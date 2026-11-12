"""Exact parse-function excerpt from the historical scorer; no API calls or configuration loading.
The original 3B extraction route remains incompletely archived. Retained top-eight rounded
alternatives do not reconstruct the complete original top-20 distribution.
Original llm_predict.py SHA256: b3f4486630bf5cb278064b4421059ec4c687819253cf66813c5c9764815c2307
"""
import math
import re

def parse(choice):
    """取 P(报错)。两条路:

    有 logprobs(旧端点)时从首 token 的概率分布算连续分;
    没有(Kimi 端点不支持)时直接用正文 Y/N,score 退化为 1.0/0.0 二值 ——
    这时 AUC 在数值上等于平衡准确率,报告里必须写明,不能和连续分混着说。

    返回 (score, answer, note)。score 为 None 表示这条不可评分 —— 记为 null,
    绝不当作"预测错了"计入指标。
    """
    lp = (choice or {}).get('logprobs') or {}
    toks = lp.get('content') or []
    content = ((choice or {}).get('message') or {}).get('content') or ''

    # !! 思考模型的 logprobs 流装的是"思考文本"的第一个 token,不是答案。
    #    实测 qwen3.8:27b 给出 'The'(思考开头 "The code..."),而 message.content
    #    明明是干净的 'N'。若不判断就会把 99.7% 的有效答复记成"不可解析"。
    #    非思考模型(qwen2.5 系列)的 logprobs[0] 就是答案 token,走下面的连续分。
    if not toks or (toks[0].get('token') or '').strip().upper() not in ('Y', 'N'):
        s = content.strip().upper()
        if s.startswith('Y'):
            return 1.0, 'Y', None
        if s.startswith('N'):
            return 0.0, 'N', None
        m = re.search(r'(?<![A-Za-z])([YN])(?![A-Za-z])', s)
        if m:
            return (1.0 if m.group(1) == 'Y' else 0.0), m.group(1), '从正文提取'
        if not s:
            return None, None, '正文为空(思考吃光预算?)'
        return None, s[:12], '答复不是 Y/N(得到 %r)' % content[:40]
    t0 = toks[0]
    py = pn = None
    for c in (t0.get('top_logprobs') or []):
        s = (c.get('token') or '').strip().upper()
        try:
            p = math.exp(c['logprob'])
        except (KeyError, OverflowError, ValueError):
            continue
        if s == 'Y' and py is None:
            py = p
        elif s == 'N' and pn is None:
            pn = p
    s0 = (t0.get('token') or '').strip().upper()
    if s0 == 'Y' and py is None:
        py = math.exp(t0['logprob'])
    if s0 == 'N' and pn is None:
        pn = math.exp(t0['logprob'])
    # 模型很自信时,对立选项根本不会出现在 top_logprobs 里 ——
    # 实测 40 条里有 7 条因此被误判为不可解析,而它们的 content 就是干净的 'Y'/'N'。
    # 这种情形按饱和处理(只找到哪个就取哪个),不算失败。
    if py is None and pn is None:
        return None, s0, '首 token 不是 Y/N(得到 %r)' % (t0.get('token'),)
    if pn is None:
        return 1.0, 'Y', None
    if py is None:
        return 0.0, 'N', None
    return py / (py + pn), s0, None
