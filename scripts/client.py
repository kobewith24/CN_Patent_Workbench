import json
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, title="自动验收工作区", url="http://127.0.0.1:8088"):
        self.url=url
        token=(ROOT/"data/access.token").read_text().strip()
        self.headers={"Authorization":f"Bearer {token}"}
        w=self.call("POST","/workspaces",{"title":title})
        self.headers.update({"X-Workspace-ID":w["id"],"X-Workspace-Key":w["key"]})
        self.workspace=w

    def raw(self,method,path,body=None,**kwargs):
        return requests.request(method,self.url+"/api"+path,headers=self.headers,json=body,timeout=180,**kwargs)

    def call(self,method,path,body=None):
        r=self.raw(method,path,body)
        if r.status_code>=400:
            raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:300]}")
        return r.json()

    def wait(self,id,timeout=180):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            j=self.call("GET",f"/jobs/{id}")
            if j["status"] not in ("queued","running","auditing"):
                return j
            time.sleep(.2)
        raise TimeoutError("Job did not finish")
