"""把 2021 的运行日志压成一行一次运行。

结构(实测):
  Action='r'  运行头,Output 全空,OutputDestination 也空
  Action='o'  输出正文,OutputDestination ∈ {stdout, stderr, system}
              system 行装的是 IDE 写的 "Process finished with exit code N"

**标签必须用退出码,不能用 traceback。** 实测:有一批运行 exit code=1 却没有任何
stderr 正文(traceback 丢失),只靠"输出里有没有 Traceback"会把这批判成"成功"。
退出码是 IDE 自己报的,与 2019 的 X-RunHasError 同一性质,是这里唯一可靠的标签源。

四类结局,绝不合并:
  ok       exit 0 且无 traceback        —— 跑通了
  err      exit 1 或出现 traceback      —— 报错
  killed   exit -1 / 130 / 大负数       —— 被用户或系统终止(2019 的 X-RunUserTerminated)
  unknown  没有 system 行或输出全空     —— **未知,不是成功**
"""
import csv
import json
import re
import sys

csv.field_size_limit(1 << 30)

TRACE = re.compile(r'Traceback \(most recent call last\)')
EXIT = re.compile(r'Process finished with exit code (-?\d+)')
ERRTAIL = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_.]*Error|[A-Za-z_][A-Za-z0-9_.]*Exception'
                     r'|KeyboardInterrupt|SystemExit|StopIteration|Terminator)\s*:', re.M)

TAIL_KEEP = 8000
HEAD_KEEP = 400
OUT = 'runs2021.jsonl'

KILLED = {-1, 130, 137, 143}


def close(cur, out, st):
    if cur is None:
        return
    tb = cur['_tb']
    ec = cur['exit_code']
    if ec is None:
        cls = 'unknown' if cur['n_out'] == 0 else 'err' if tb else 'unknown'
    elif ec in KILLED:
        cls = 'killed'
    elif ec == 0:
        cls = 'err' if tb else 'ok'
    else:
        cls = 'err'
    cur['outcome'] = cls
    cur['traceback'] = tb
    if tb:
        m = [x.group(1) for x in ERRTAIL.finditer(cur['_tail'])]
        cur['err_type'] = m[-1] if m else None
    cur['out_head'] = cur['_head']
    cur['out_tail'] = cur['_tail'] if (tb or cls == 'err') else None
    for k in ('_head', '_tail', '_tb'):
        cur.pop(k)
    out.write(json.dumps(cur, ensure_ascii=False) + '\n')
    st[cls] += 1
    st['runs'] += 1


def main():
    st = {'rows': 0, 'runs': 0, 'ok': 0, 'err': 0, 'killed': 0, 'unknown': 0}
    cur = None
    with open(OUT, 'w', encoding='utf-8') as out:
        for path in ('cs1/runs1.csv', 'cs1/runs2.csv'):
            with open(path, encoding='utf-8', newline='') as fh:
                r = csv.reader(fh)
                head = next(r)
                ix = {c: i for i, c in enumerate(head)}
                iS, iA, iF = ix['SubjectID'], ix['AssignmentID'], ix['File']
                iAct, iR = ix['Action'], ix['RunID']
                iO, iT, iD = ix['Output'], ix['ClientTimestamp'], ix['OutputDestination']
                for row in r:
                    st['rows'] += 1
                    if st['rows'] % 2000000 == 0:
                        print('  ... %d 行,已收 %d 次(ok %d err %d killed %d unk %d)'
                              % (st['rows'], st['runs'], st['ok'], st['err'],
                                 st['killed'], st['unknown']), flush=True)
                    a = row[iAct]
                    if a == 'r':
                        close(cur, out, st)
                        cur = {'RunID': row[iR], 'SubjectID': row[iS],
                               'AssignmentID': row[iA], 'File': row[iF],
                               'ts': row[iT], 'n_out': 0, 'exit_code': None,
                               'outcome': None, 'traceback': False, 'err_type': None,
                               'out_head': None, 'out_tail': None,
                               '_head': '', '_tail': '', '_tb': False}
                    elif a == 'o' and cur is not None:
                        c, d = row[iO], row[iD]
                        cur['n_out'] += 1
                        if len(cur['_head']) < HEAD_KEEP:
                            cur['_head'] += c[:HEAD_KEEP - len(cur['_head'])]
                        cur['_tail'] = (cur['_tail'] + '\n' + c)[-TAIL_KEEP:]
                        if not cur['_tb'] and TRACE.search(c):
                            cur['_tb'] = True
                        if d == 'system':
                            m = EXIT.search(c)
                            if m:
                                cur['exit_code'] = int(m.group(1))
        close(cur, out, st)      # 必须在 with 内 —— 出了块文件就关了

    print('\n扫了 %d 行,得到 %d 次运行 -> %s' % (st['rows'], st['runs'], OUT))
    n = max(st['runs'], 1)
    for k, desc in (('ok', '跑通 exit 0 无 traceback'),
                    ('err', '报错 exit≠0 或有 traceback'),
                    ('killed', '被终止 exit -1/130'),
                    ('unknown', '未知(无退出码且无输出)')):
        print('  %-8s %7d (%.1f%%)   %s' % (k, st[k], 100.0 * st[k] / n, desc))
    known = st['ok'] + st['err']
    if known:
        print('  可用的两类里,报错率 %.1f%%' % (100.0 * st['err'] / known))


if __name__ == '__main__':
    main()
