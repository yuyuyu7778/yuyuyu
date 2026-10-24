import re, zlib, sys

def streams(data):
    out=[]
    for m in re.finditer(rb'stream\r?\n', data):
        start=m.end(); e=data.find(b'endstream', start)
        try: out.append(zlib.decompress(data[start:e]))
        except Exception: pass
    return out

def parse_cmaps(chunks):
    m={}
    for c in chunks:
        if b'beginbfchar' not in c and b'beginbfrange' not in c: continue
        for blk in re.findall(rb'beginbfchar(.*?)endbfchar', c, re.S):
            for a,b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                try: m[int(a,16)]=bytes.fromhex(b.decode()).decode('utf-16-be','ignore')
                except Exception: pass
        for blk in re.findall(rb'beginbfrange(.*?)endbfrange', c, re.S):
            for lo,hi,dst in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                lo,hi,base=int(lo,16),int(hi,16),int(dst,16)
                for i in range(lo,hi+1):
                    ch=base+i-lo
                    if ch<0x110000: m[i]=chr(ch)
    return m

def extract(path):
    data=open(path,'rb').read(); chunks=streams(data); merged=parse_cmaps(chunks)
    texts=[]
    for c in chunks:
        if b'BT' not in c or b'beginbfchar' in c or b'beginbfrange' in c: continue
        page=[]
        for bt in re.findall(rb'BT(.*?)ET', c, re.S):
            for hx in re.findall(rb'<([0-9A-Fa-f]+)>\s*Tj', bt):
                s=hx.decode()
                page.append(''.join(merged.get(int(s[i:i+4],16),'?') for i in range(0,len(s),4)))
        if page: texts.append(''.join(page))
    return '\n\n=== PAGE BREAK ===\n\n'.join(texts)

if __name__=='__main__':
    for name,p in [('A','2026年MathorCup大数据挑战赛-赛道A初赛/2026年MathorCup大数据挑战赛-赛道A初赛.pdf'),('B','2026年MathorCup大数据挑战赛-赛道B初赛/2026年MathorCup大数据挑战赛-赛道B初赛.pdf')]:
        t=extract(p)
        open('pdf_%s.txt'%name,'w',encoding='utf-8').write(t)
        print(name,len(t))
