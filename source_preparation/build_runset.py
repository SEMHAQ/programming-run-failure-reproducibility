"""建配对数据集:(运行时刻的代码状态, 运行结果, 过程特征)。

真值来源:X-RunHasError —— 学生程序真实执行的结果,自动产生,零人工评分。
代码状态:用 RECONSTRUCTION.md 里验证过的重建逻辑(逐事件保真度 99.712%)。

**只输出有执行记录的事件。** 2019 的春季一个执行结果都没采集
(见 RUN_OUTCOMES.md),所以这个数据集天然是秋季的。

输出:
  runset.jsonl  每条一个 JSON,含完整代码 —— 给"LLM 读代码"那一路用
  runset.csv    去掉代码列的特征表 —— 给过程特征模型用
"""
import ast
import csv
import json
import sys

csv.field_size_limit(10 ** 9)

SRC = sys.argv[1] if len(sys.argv) > 1 else 'cs1_2019/keystrokes.csv'
OUT_J = sys.argv[2] if len(sys.argv) > 2 else 'runset.jsonl'
OUT_C = sys.argv[3] if len(sys.argv) > 3 else 'runset.csv'
CODE_CAP = 50000          # 超过就截断并标记,避免个别超大缓冲区撑爆文件

EDIT_TYPES = ('Insert', 'Paste', 'X-Compose', 'Redo', 'Delete', 'Undo', 'Cut')


def apply_edit(buf, et, i, ins, dele):
    if et == 'X-Checkpoint':
        return ins
    if i < 0:
        return None
    if et in EDIT_TYPES:
        return buf[:i] + ins + buf[i + len(dele):]
    return None


def compiles(s):
    try:
        ast.parse(s)
        return True
    except (SyntaxError, ValueError):
        return False


FIELDS = ['EventID', 'SubjectID', 'AssignmentID', 'CodeStateSection', 'X-Task',
          'ClientTimestamp', 'X-Compilable', 'X-RunHasError',
          'X-RunUserTerminated', 'X-RunInput', 'X-RunOutput',
          'buf_len', 'n_lines', 'n_other_sec', 'other_sec_compiles',
          'eidx', 'blk_run_idx', 'edits_since_last_run', 'secs_since_last_run',
          'edits_since_clean', 'secs_since_last_edit',
          'blk_prior_errs', 'subset']

n_out = 0
n_spring = 0
n_trunc = 0
n_empty = 0

with open(SRC, encoding='utf-8', errors='replace', newline='') as fh, \
        open(OUT_J, 'w', encoding='utf-8') as fj, \
        open(OUT_C, 'w', encoding='utf-8', newline='') as fc:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: i for i, c in enumerate(hdr)}
    need = max(ix.values())
    iS, iA, iSec = ix['SubjectID'], ix['AssignmentID'], ix['CodeStateSection']
    iET, iC, iH = ix['EventType'], ix['X-Compilable'], ix['X-RunHasError']
    iO, iI, iU = ix['X-RunOutput'], ix['X-RunInput'], ix['X-RunUserTerminated']
    iE, iL, iD, iT = (ix['EventID'], ix['SourceLocation'],
                      ix['DeleteText'], ix['EditType'])
    # 必须是独立的变量名。曾把 InsertText 写成上面的 iI(= X-RunInput),
    # 而 X-RunInput 在编辑事件上恒为空串 —— 于是每次插入都插入空字符串,
    # 缓冲区全长 0,且不报任何错。这类"静默为空"只能靠下游断言抓。
    iN = ix['InsertText']
    iTS, iTask = ix['ClientTimestamp'], ix['X-Task']

    w = csv.writer(fc)
    w.writerow(FIELDS)

    key = None
    bufs = {}
    blk_eidx0 = 0
    last_clean = {}
    last_run_eidx = last_run_ts = None
    last_edit_ts = None
    blk_runs = blk_errs = 0
    eidx = 0

    for p in rd:
        if len(p) <= need:
            continue
        k = (p[iS], p[iA])
        sec = p[iSec]

        if k != key:
            key, bufs = k, {}
            last_clean = {}
            blk_eidx0 = eidx
            last_run_eidx = last_run_ts = None
            last_edit_ts = None
            blk_runs = blk_errs = 0

        et = p[iT]

        if p[iET] == 'Run.Program':
            # 只收有执行记录的 —— 春季那些没有记录,不是"跑成功了"
            if p[iH] == '':
                n_spring += 1
                continue

            buf = bufs.get(sec, '')
            others = [s for s in bufs if s != sec]
            try:
                ts = int(float(p[iTS]))
            except ValueError:
                ts = None

            run_err = p[iH] == 'True'
            row = {
                'EventID': p[iE], 'SubjectID': p[iS], 'AssignmentID': p[iA],
                'CodeStateSection': sec, 'X-Task': p[iTask],
                'ClientTimestamp': p[iTS],
                'X-Compilable': p[iC], 'X-RunHasError': p[iH],
                'X-RunUserTerminated': p[iU],
                'X-RunInput': p[iI], 'X-RunOutput': p[iO],
                'buf_len': len(buf), 'n_lines': buf.count('\n') + 1,
                'n_other_sec': len(others),
                'other_sec_compiles': int(any(compiles(bufs[s]) for s in others))
                                      if others else -1,
                'eidx': eidx, 'blk_run_idx': blk_runs,
                'edits_since_last_run': (eidx - last_run_eidx)
                                        if last_run_eidx is not None else -1,
                'secs_since_last_run': round((ts - last_run_ts) / 1000.0, 1)
                                       if (ts and last_run_ts) else -1,
                'edits_since_clean': eidx - last_clean.get(sec, blk_eidx0),
                'secs_since_last_edit': round((ts - last_edit_ts) / 1000.0, 1)
                                        if (ts and last_edit_ts) else -1,
                'blk_prior_errs': blk_errs,
                'subset': 'fall',
                'code': buf, 'code_truncated': len(buf) > CODE_CAP,
            }
            if len(buf) > CODE_CAP:
                n_trunc += 1
                row['code'] = buf[:CODE_CAP]

            fj.write(json.dumps(row, ensure_ascii=False) + '\n')
            w.writerow([row[f] for f in FIELDS])

            n_out += 1
            if not buf:
                n_empty += 1
            blk_runs += 1
            if run_err:
                blk_errs += 1
            last_run_eidx = eidx
            last_run_ts = ts
            continue

        if not et:
            continue

        if et == 'X-Checkpoint':
            bufs[sec] = p[iN]
        else:
            raw = p[iL].strip()
            try:
                i = int(float(raw))
            except ValueError:
                i = -1
            nb = apply_edit(bufs.get(sec, ''), et, i, p[iN], p[iD])
            if nb is None:
                continue
            bufs[sec] = nb

        eidx += 1
        try:
            last_edit_ts = int(float(p[iTS]))
        except ValueError:
            pass
        # 用数据自带的 X-Compilable 记"上次干净",不要在这里对每个事件跑 ast.parse
        # —— 500 万次解析纯属浪费,而且 X-Compilable 就是记录器当时的判定。
        if p[iC] == '1':
            last_clean[sec] = eidx

print('写出 %d 条 -> %s / %s' % (n_out, OUT_J, OUT_C))
print('跳过(无执行记录的运行事件,全部来自春季) %d' % n_spring)
print('其中重建代码为空的 %d 条 (%.1f%%)' % (n_empty, 100.0 * n_empty / n_out if n_out else 0))
if n_trunc:
    print('其中代码被截断的 %d 条 (超过 %d 字符)' % (n_trunc, CODE_CAP))

# 上一次这个脚本因为 iI 绑错列,把 32,997 条代码全写成了空串,却没有任何报错。
# 之后一路把"代码长度"喂给模型,还得出"代码形态没用"的假结论。宁可炸。
if n_out and n_empty > 0.5 * n_out:
    raise SystemExit('!! 超过一半的运行事件重建出空代码 —— 几乎可以肯定是取列取错了,'
                     '不要用这份数据集。')
