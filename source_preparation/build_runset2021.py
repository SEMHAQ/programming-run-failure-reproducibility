"""建 2021 的配对数据集:(运行那一刻重建出的代码, 这次运行的真实结果)。

与 2019 的两处关键差异,决定了这里的做法:

1. **代码状态那一侧是精确的。** 2019 的运行事件内联在编辑流里;2021 的运行结果
   在另一张表,但 keystrokes.csv 里仍有 Run.Program 事件,它在编辑流中有确切位置。
   所以我们在 Run.Program 上快照缓冲区 —— 位置不用猜。只有**标签**那一侧要连接。

2. **标签是推断来的,而且连接只有部分重叠。** runs.csv 里没有报错列,只有 Output
   正文;靠 traceback 推断(见 extract_runs2021.py)。而且两张表的运行只有部分
   重合(54,561 vs 24,195),时间戳尾部差到几天 —— 那是"对不上的在跟一次无关运行
   配对",不是时钟偏差。所以**默认只认精确相等**,并把匹配质量逐条记下来供收紧。

标签三态:
  1 = 输出里有 Python traceback
  0 = 有输出、无 traceback
  ''= 输出全空 —— **未知,不是无错**,绝不并入 0

用法:python build_runset2021.py
"""
import ast
import bisect
import csv
import json
from collections import defaultdict

csv.field_size_limit(1 << 30)

KS = 'cs1/keystrokes.csv'
RUNS = 'runs2021.jsonl'
OUT_CSV = 'runset2021.csv'
OUT_JSONL = 'runset2021.jsonl'

TOL = 1000          # 候选计数窗口(毫秒)

META = ['buf_len', 'n_lines', 'eidx', 'blk_run_idx', 'edits_since_last_run',
        'secs_since_last_run', 'edits_since_clean', 'secs_since_last_edit',
        'n_other_sec', 'other_sec_compiles']

COLS = ['RunKey', 'EventID', 'SubjectID', 'AssignmentID', 'CodeStateSection',
        'ts', 'X-Compilable', 'label', 'outcome', 'match', 'match_delta', 'n_cand',
        'n_out', 'err_type', 'buf_parses'] + META


def new_state():
    return {'eidx': 0, 'blk_run_idx': 0, 'edits_since_last_run': 0,
            'secs_since_last_run': -1, 'edits_since_clean': 0,
            'secs_since_last_edit': -1, 'buf_len': 0, 'n_lines': 0,
            '_last_run_ts': None, '_last_edit_ts': None}


def main():
    runs = {}
    with open(RUNS, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            try:
                t = int(float(r['ts']))
            except (TypeError, ValueError):
                continue
            runs[(r['SubjectID'], r['AssignmentID'], r['File'], t)] = r
    print('标签表 %d 条' % len(runs))

    bygrp = defaultdict(list)
    for (s, a, f, t) in runs:
        bygrp[(s, a, f)].append(t)
    for k in bygrp:
        bygrp[k].sort()

    buf = {}
    state = {}
    last_sec = {}
    recs = []
    n_rows = n_run = n_warn = 0

    with open(KS, encoding='utf-8', newline='') as fh:
        rd = csv.reader(fh)
        head = next(rd)
        ix = {c: i for i, c in enumerate(head)}
        iE, iS, iA, iSec = (ix['EventID'], ix['SubjectID'], ix['AssignmentID'],
                            ix['CodeStateSection'])
        iT, iL, iD = ix['EventType'], ix['SourceLocation'], ix['EditType']
        iI, iDel = ix['InsertText'], ix['DeleteText']
        iTs, iC = ix['ClientTimestamp'], ix['X-Compilable']

        for row in rd:
            n_rows += 1
            if n_rows % 500000 == 0:
                print('  ... %d 行,快照 %d 次' % (n_rows, n_run), flush=True)
            key = (row[iS], row[iA], row[iSec])
            blk = (row[iS], row[iA])

            # ---------------- 运行时刻:先取量,再更新游标 ----------------
            if row[iT] == 'Run.Program':
                try:
                    ts = int(float(row[iTs]))
                except ValueError:
                    continue
                st = state.get(key)
                if st is None:                 # 该文件还没有过编辑事件
                    n_warn += 1
                    continue

                grp = bygrp.get((row[iS], row[iA], row[iSec]), [])
                j = bisect.bisect_left(grp, ts)
                best = None
                for jj in (j - 1, j):
                    if 0 <= jj < len(grp):
                        d = abs(grp[jj] - ts)
                        if best is None or d < best:
                            best = d
                n_cand = 0
                if grp:
                    lo = bisect.bisect_left(grp, ts - TOL)
                    hi = bisect.bisect_right(grp, ts + TOL)
                    n_cand = hi - lo
                r = runs.get((row[iS], row[iA], row[iSec], ts))
                # 标签用退出码口径(见 extract_runs2021.py):只有 ok/err 两类可用,
                # killed(被终止)与 unknown(无退出码)一律记为缺失,不并入 0。
                occ = r.get('outcome') if r else None
                label = {'ok': 0, 'err': 1}.get(occ)

                others = [v for s, v in last_sec.get(blk, {}).items() if s != row[iSec]]
                code = buf.get(key, '')
                try:
                    ast.parse(code)
                    parses = 1
                except (SyntaxError, ValueError, MemoryError, RecursionError):
                    parses = 0

                rec = {'RunKey': '%s|%s|%s|%d' % (row[iS], row[iA], row[iSec], ts),
                       'EventID': row[iE], 'SubjectID': row[iS],
                       'AssignmentID': row[iA], 'CodeStateSection': row[iSec],
                       'ts': ts, 'X-Compilable': row[iC],
                       'label': '' if label is None else label,
                       'match': 'exact' if r else ('none' if best is None else 'near'),
                       'match_delta': '' if best is None else best,
                       'n_cand': n_cand,
                       'n_out': r.get('n_out') if r else '',
                       'err_type': (r.get('err_type') or '') if r else '',
                       'outcome': (occ or '') if r else '',
                       'buf_parses': parses}
                for m in ('buf_len', 'n_lines', 'eidx', 'blk_run_idx',
                          'edits_since_last_run', 'edits_since_clean'):
                    rec[m] = st[m]
                rec['secs_since_last_run'] = (-1 if st['_last_run_ts'] is None
                                              else max(0, (ts - st['_last_run_ts']) // 1000))
                rec['secs_since_last_edit'] = (-1 if st['_last_edit_ts'] is None
                                               else max(0, (ts - st['_last_edit_ts']) // 1000))
                rec['n_other_sec'] = len(others)
                rec['other_sec_compiles'] = ('' if not others
                                             else ('1' if any(x == '1' for x in others)
                                                   else '0'))
                rec['_code'] = code
                recs.append(rec)
                n_run += 1

                # 更新游标(必须在取量之后)
                st['blk_run_idx'] += 1
                st['edits_since_last_run'] = 0
                st['_last_run_ts'] = ts
                continue

            # ---------------- 编辑事件 ----------------
            ed = row[iD]
            if not ed:
                continue
            try:
                loc = int(float(row[iL]))
            except ValueError:
                continue
            b = buf.get(key, '')
            if loc > len(b):
                continue
            nb = b[:loc] + row[iI] + b[loc + len(row[iDel]):]
            buf[key] = nb

            st = state.get(key)
            if st is None:
                st = state[key] = new_state()
            st['eidx'] += 1
            st['edits_since_last_run'] += 1
            st['edits_since_clean'] = 0 if row[iC] == '1' else st['edits_since_clean'] + 1
            st['buf_len'] = len(nb)
            st['n_lines'] = nb.count('\n') + 1
            st['secs_since_last_edit'] = 0
            try:
                st['_last_edit_ts'] = int(float(row[iTs]))
            except ValueError:
                pass
            if row[iC] in ('0', '1'):
                last_sec.setdefault(blk, {})[row[iSec]] = row[iC]

    print('\n扫描 %d 行,快照 %d 次(该文件无编辑事件而跳过 %d)'
          % (n_rows, n_run, n_warn))

    if not recs:
        raise SystemExit('!! 一条快照都没有')
    n_empty = sum(1 for r in recs if not r['_code'])
    if n_empty > 0.5 * len(recs):
        raise SystemExit('!! 过半快照是空代码 —— 几乎肯定是取列取错了,不要用这份数据集。')

    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for r in recs:
            w.writerow(r)
    with open(OUT_JSONL, 'w', encoding='utf-8') as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    n_ex = sum(1 for r in recs if r['match'] == 'exact')
    n_lab = sum(1 for r in recs if r['label'] != '')
    print('写出 %s / %s' % (OUT_CSV, OUT_JSONL))
    print('  精确匹配 %d (%.1f%%)   其中有标签 %d'
          % (n_ex, 100.0 * n_ex / len(recs), n_lab))
    print('  快照代码为空 %d (%.2f%%)' % (n_empty, 100.0 * n_empty / len(recs)))


if __name__ == '__main__':
    main()
