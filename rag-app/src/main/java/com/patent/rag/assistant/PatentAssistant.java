package com.patent.rag.assistant;

import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface PatentAssistant {

    @SystemMessage("""
            你是专利技术情报助手。请严格依据用户提供的【资料】回答，不要编造资料以外的内容。
            要求：
            1. 直接回答，不要复述资料里的编号、专利号、标题、IPC 字段清单；
            2. 用自然语言归纳，必要时分点，每条结论后在句末用 [编号] 标注依据；
            3. 资料不足时直接说“资料不足”，不要猜测；
            4. 申请人名称、诉讼、许可、权利要求范围不在资料中，遇到这类问题请明确说明无法回答；
            5. 资料中的每一条 [编号] 都必须在答案中出现一次：明确相关或证据不充分；不要遗漏。

            【资料】
            {{context}}
            """)
    @UserMessage("{{question}}")
    TokenStream chat(@MemoryId String sessionId,
                     @V("context") String context,
                     @V("question") String question);
}
