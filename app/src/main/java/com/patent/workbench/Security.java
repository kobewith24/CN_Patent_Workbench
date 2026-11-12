package com.patent.workbench;

import jakarta.servlet.*;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class Security extends OncePerRequestFilter {
  private final Settings settings;
  private final Store store;

  public Security(Settings settings, Store store) {
    this.settings = settings;
    this.store = store;
  }

  @Override
  protected void doFilterInternal(
      HttpServletRequest req, HttpServletResponse res, FilterChain chain)
      throws ServletException, IOException {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "no-referrer");
    res.setHeader(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src"
            + " 'self' data:; frame-ancestors 'none'; base-uri 'self'");
    String path = req.getRequestURI();
    if (!path.startsWith("/api/") || path.equals("/api/health")) {
      chain.doFilter(req, res);
      return;
    }
    String token = req.getHeader("Authorization");
    if (token == null
        || !token.startsWith("Bearer ")
        || !MessageDigest.isEqual(
            token.substring(7).getBytes(StandardCharsets.UTF_8),
            settings.apiToken.getBytes(StandardCharsets.UTF_8))) {
      deny(res, 401, "访问令牌无效");
      return;
    }
    if (path.equals("/api/workspaces") && req.getMethod().equals("POST")) {
      chain.doFilter(req, res);
      return;
    }
    try {
      if (!store.workspaceValid(
          req.getHeader("X-Workspace-ID"), req.getHeader("X-Workspace-Key"))) {
        deny(res, 403, "工作区凭据无效");
        return;
      }
    } catch (Exception e) {
      deny(res, 503, "工作区存储暂不可用");
      return;
    }
    req.setAttribute("workspace", req.getHeader("X-Workspace-ID"));
    chain.doFilter(req, res);
  }

  private void deny(HttpServletResponse res, int code, String message) throws IOException {
    res.setStatus(code);
    res.setContentType("application/json;charset=UTF-8");
    res.getWriter().write("{\"message\":\"" + message + "\"}");
  }
}
