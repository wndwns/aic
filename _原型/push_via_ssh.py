#!/usr/bin/env python
"""经 SSH(443) + 本地 SOCKS5 代理推送 GitHub —— 绕开 HTTPS 的 workflow scope 限制。

背景：本机 HTTPS 凭据（GCM / gh 的 gho_ token）都没有 `workflow` scope，
只要提交里含 `.github/workflows/*` 就会被 GitHub 拒：

    refusing to allow an OAuth App to create or update workflow ... without 'workflow' scope

SSH 认证不受该限制，但本机「直连不通」：`git@github.com:22` 与 `ssh.github.com:443`
都被墙挡掉，且 Git Bash 没有 nc / ncat / connect / socat 可做 ProxyCommand。
本脚本自带 SOCKS5 隧道（Clash 7897 支持 SOCKS5），无需任何外部依赖。

用法：

    python _原型/push_via_ssh.py --check                     # 只验认证，不动远端
    python _原型/push_via_ssh.py feature/demo-guide          # 推工作分支
    python _原型/push_via_ssh.py feature/demo-guide:main     # 快进推 main（可多条）

    # 被 ssh 当 ProxyCommand 调用（脚本内部使用，不用手敲）：
    python _原型/push_via_ssh.py --bridge <host> <port>

前提：Clash 在跑（默认 127.0.0.1:7897 为混合端口，同时支持 HTTP 与 SOCKS5）。
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

DEFAULT_PROXY = "127.0.0.1:7897"
SSH_HOST = "ssh.github.com"
SSH_PORT = 443
REMOTE_NAME = "origin"


def repo_ssh_url(explicit: str | None = None) -> str:
    """把 origin 的地址转成「SSH over 443」形式。

    以前这里把仓库路径写死成 `wndwns/-.git`，导致同一个项目改投别的比赛、
    换到新仓库后，脚本仍然往旧仓库推。改成从 origin 推导，换仓库不用改脚本。
    """
    if explicit:
        raw = explicit
    else:
        raw = subprocess.run(["git", "remote", "get-url", REMOTE_NAME],
                             capture_output=True, text=True).stdout.strip()
    if not raw:
        raise SystemExit(f"取不到 {REMOTE_NAME} 的地址，请先 git remote set-url")

    path = raw
    for prefix in ("https://github.com/", "http://github.com/",
                   "ssh://git@github.com/", f"ssh://git@{SSH_HOST}:{SSH_PORT}/",
                   "git@github.com:"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    path = path.strip("/")
    if not path.endswith(".git"):
        path += ".git"
    return f"ssh://git@{SSH_HOST}:{SSH_PORT}/{path}"


def _force_ref(branch: str, sha: str) -> bool:
    """写分支引用并回读校验。

    本机会出现 `git update-ref` / `git branch -f` 返回成功却不落盘的情况，
    所以写后必须回读；不一致就手写 ref 文件（实测有效）。
    """
    subprocess.call(["git", "update-ref", f"refs/heads/{branch}", sha],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    got = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
                         capture_output=True, text=True).stdout.strip()
    if got == sha:
        return True
    ref_file = Path(".git") / "refs" / "heads" / branch
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_text(sha + "\n", encoding="ascii")
    got = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
                         capture_output=True, text=True).stdout.strip()
    return got == sha


def _proxy_args(proxy: str) -> list[str]:
    """git 的 HTTPS 请求走同一个 SOCKS5 代理。

    本机直连 github:443 不通（Clash 的 HTTP 隧道会回 502），
    所以 fetch / ls-remote 这类非 SSH 操作也必须显式指代理，否则会静默失败。
    """
    return ["-c", f"http.proxy=socks5h://{proxy}", "-c", f"https.proxy=socks5h://{proxy}"]


def sync_local_refs(proxy: str) -> None:
    """按**远端真值**对齐本地分支。

    ⚠️ 不读 `origin/<branch>` 这类 remote-tracking 引用：本机出现过 packed-refs
    残留陈旧值把本地 main 带歪的情况，所以直接问远端。
    """
    out = subprocess.run(["git", *_proxy_args(proxy), "ls-remote", REMOTE_NAME],
                         capture_output=True, text=True).stdout
    if not out.strip():
        print("  （远端引用读取不到，跳过本地对齐；请手工 git ls-remote 核对）")
        return
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        if not ref.startswith("refs/heads/"):
            continue
        branch = ref[len("refs/heads/"):]
        cur = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
                             capture_output=True, text=True)
        if cur.returncode != 0:
            continue          # 本地没这个分支，不动
        if cur.stdout.strip() == sha:
            continue
        ok = _force_ref(branch, sha)
        print(f"  {branch}: 本地对齐到 {sha[:7]}{'' if ok else '（写入失败，请手工核对）'}")


# --------------------------------------------------------------------- SOCKS5 桥

def bridge(host: str, port: int, proxy_host: str, proxy_port: int) -> int:
    """把 stdin/stdout 与「经 SOCKS5 代理连到 host:port」的套接字双向对接。"""
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    sock.sendall(b"\x05\x01\x00")                       # 只声明「无认证」
    ver, method = sock.recv(2)
    if ver != 5 or method != 0:
        print(f"代理拒绝握手: ver={ver} method={method}", file=sys.stderr)
        return 1
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host.encode("idna")
                 + struct.pack("!H", port))
    resp = sock.recv(10)
    if len(resp) < 2 or resp[1] != 0:
        print(f"代理 CONNECT 失败: reply={resp!r}", file=sys.stderr)
        return 1

    def pump_stdin() -> None:
        try:
            while True:
                chunk = sys.stdin.buffer.read1(65536)
                if not chunk:
                    break
                sock.sendall(chunk)
        except Exception:
            pass
        finally:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    threading.Thread(target=pump_stdin, daemon=True).start()
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    except Exception:
        pass
    return 0


# ------------------------------------------------------------------ 推送驱动

def write_ssh_config(proxy: str) -> str:
    """生成临时 ssh config：ProxyCommand 指回本脚本的 --bridge 模式。"""
    proxy_host, _, proxy_port = proxy.partition(":")
    me = os.path.abspath(__file__)
    known_hosts = os.path.join(tempfile.gettempdir(), "gh_push_known_hosts").replace("\\", "/")
    cfg = (
        f"Host {SSH_HOST}\n"
        f"  HostName {SSH_HOST}\n"
        f"  Port {SSH_PORT}\n"
        f"  User git\n"
        f'  ProxyCommand "{sys.executable.replace(chr(92), "/")}" "{me.replace(chr(92), "/")}"'
        f" --bridge %h %p --proxy {proxy_host}:{proxy_port or '7897'}\n"
        f"  StrictHostKeyChecking accept-new\n"
        f"  UserKnownHostsFile {known_hosts}\n"
    )
    # 正斜杠：这个路径要嵌进 `git -c core.sshCommand="ssh -F <path>"`，
    # 那串会经 sh 解析，反斜杠会被当转义吃掉（实测变成 C:UsersWHAppData...）。
    path = os.path.join(tempfile.gettempdir(), "gh_push_ssh_config").replace("\\", "/")
    # 必须 UTF-8：本项目路径含中文（项目 / 副本），ASCII 会直接抛 UnicodeEncodeError
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cfg)
    return path


def git(args: list[str], ssh_config: str) -> int:
    env = dict(os.environ)
    env.pop("GIT_SSH_COMMAND", None)
    cmd = ["git", "-c", f"core.sshCommand=ssh -F {ssh_config}", *args]
    print("  $ git " + " ".join(args), flush=True)
    return subprocess.call(cmd, env=env)


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    proxy = DEFAULT_PROXY
    if "--proxy" in args:
        i = args.index("--proxy")
        proxy = args[i + 1]
        del args[i:i + 2]

    explicit_url = None
    if "--repo-url" in args:
        i = args.index("--repo-url")
        explicit_url = args[i + 1]
        del args[i:i + 2]

    if args[:1] == ["--bridge"]:
        host, port = args[1], int(args[2])
        ph, _, pp = proxy.partition(":")
        return bridge(host, port, ph, int(pp or 7897))

    ssh_config = write_ssh_config(proxy)
    url = repo_ssh_url(explicit_url)

    if not args or args == ["--check"]:
        print(f"验证 SSH 认证（经 {proxy} → {SSH_HOST}:{SSH_PORT}）…")
        print(f"目标仓库：{url}")
        rc = subprocess.call(["ssh", "-F", ssh_config, "-T", f"git@{SSH_HOST}"],
                             stderr=subprocess.STDOUT)
        return 0 if rc == 0 else rc

    use_remote = "--raw" not in args
    for refspec in [a for a in args if a != "--raw"]:
        if use_remote:
            print(f"  $ git push {url} {refspec}")
            rc = git(["push", url, refspec], ssh_config)
        else:
            rc = git(["push", refspec], ssh_config)
        if rc != 0:
            print(f"\n❌ 推送失败（{refspec}），退出码 {rc}", file=sys.stderr)
            return rc

    print("\n本地引用同步中…")
    subprocess.call(["git", *_proxy_args(proxy), "fetch", REMOTE_NAME],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sync_local_refs(proxy)
    print("✅ 推送完成，本地分支已按远端真值对齐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
