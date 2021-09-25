import time
t1=time.time()
t0=time.time()

import itertools as it
import numpy as np
import math
import pickle
import copy
from functools import reduce
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def indices(A):
    return range(len(A))

def M(x):#x(ベクトル表示)の基本行列表示の逆行列(転置)
    #l=len(x)
    #return np.asarray([[Cdelta(j,x[i])for i in range(l)] for j in range(l)])
    return np.identity(len(x), dtype=int)[:, x]

def inv(A):#行列式1の行列の逆行列(転置行列)
    #l=len(A)
    #return np.asarray([[A[i][j] for i in range(l)] for j in range(l)])
    return np.swapaxes(A, -1, -2)



def iM(x):
    return inv(M(x))

def get_time():
    t=time.time()
    dt=(t-t0)
    print("合計時間:",dt)

def finish():
    t=time.time()
    dt=(t-t0)
    print("合計時間:",dt)
    exit()

def cycle(v):#ベクトル表示→サイクル表示
    rest=[i for i in indices(v)]
    cv=[]
    while len(rest)!=0:
        cvi=[rest[0]]
        rest.remove(rest[0])
        #print(rest)
        j=0
        while True:
            cvi.append(v[cvi[j]])
            try:
                rest.remove(v[cvi[j]])#無いものをremoveできない
            except ValueError:
                pass
            j=j+1
            #print(cvi)
            if cvi[j]==cvi[0]:
                cvi.pop(j)
                break
        cv.append(cvi)
    return cv#各成分の長さが違うので抽出後adarrayにしないといけない

def icycle(c,L="no data"):#サイクル表示→ベクトル表示
    if L=="no data":L=sum([len(c[i]) for i in indices(c)])
    v=[i for i in range(L)]
    for i in indices(c):
        for j in range(len(c[i])-1):
            v[c[i][j]]=c[i][j+1]
        v[c[i][len(c[i])-1]]=c[i][0]
    return v


def Sym(d):
    #S=np.asarray([[[Cdelta(i,s[j])for i in range(d)] for j in range(d)] for s in list(it.permutations(list(range(d))))])
    a = np.identity(d, dtype='i')
    S = np.array([a[np.array(idx)] for idx in it.permutations(range(d))])
    return S

def represent(X,d):
    p=len(X)#5
    N=np.asarray([len(x) for x in X])#[1,2,2,1,1]
    p0=sum(N)#7
    Y=[]
    for i in range(p):#iは分けた個数
        for j in range(N[i]):
            #Xij[k]=[0] if X[i][j][k]==1 else np.concatenate([s+1 for s in range(X[i][j][k]-1)],[0])
            Xij=[ np.asarray([0]) if X[i][j][k]==1 else np.asarray([s+1 for s in range(X[i][j][k]-1)]+[0]) for k in range(i+1)]
            Dij=[0]+[len(Xij[k]) for k in range(i)] #np.asarray([0]+[len(Xij[k]) for k in range(i)])
            Eij=np.asarray([sum(Dij[0:k+1]) for k in range(i+1)])
            Yij=np.concatenate([Xij[k]+Eij[k] for k in range(i+1)])
            #print(i,Xij,Eij,Yij)
            Y.append(Yij)
    return np.asarray(Y)


########all patterns of permutation numbers in d

def Sign(d):
    return np.asarray([[(t//(2**s))%2 for s in range(d)] for t in range(2**d)])

def vinv(v):
    return M(v).dot(e)

def conjugate(sigma,x,invsigma=None):
    if invsigma is None:
        return np.array([sigma[ x[vinv(sigma)[j]] ] for j in range(d)])
    else:
        return np.array([sigma[ x[invsigma[j]] ] for j in range(d)])


#signは符号列ではなく符号値として参照する
def ytosign(j,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return Vd[j] if sign == 1 else iVd[j]

def xtosign(i,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return Xrep[i] if sign == 1 else iXrep[i]

def vtosign(v,sign):#ベクトルを参照するバージョン
    return v if sign == 1 else vinv(v)

def vtosigns(v,signs):#符号関数を参照するバージョン#符号に注意！！！！！！
    return np.array([vtosign(v,not signs[m])[m] for m in Id])

def xi(y,e):
    return np.array([e[m]!=e[vtosign(y,not e[m])[m]] for m in Id])

def isom_sub(i, j, k):
    ret=[]
    x=Xrep[i]
    y=Vd[j]
    invy=vinv(y)
    yi=iVd[j]
    eps=Signd[k]
    for nd in range(lSignd):
        delta=Signd[nd]
        vtosign_x=[vtosign(x,not delta[m])[m] for m in Id]
        if np.any(delta!=np.array([delta[x[m]] for m in Id])):
            continue
        for ns in range(lVd):
            sigma=Vd[ns]
            isigma=iVd[ns]
            invsigma = vinv(sigma)
            if np.any(conjugate(sigma,vtosign_x,invsigma)!=x):
                continue
            Yeesd=np.array([conjugate(sigma,[invy[m] if (eps[m]+Signd[n][sigma[m]]+delta[m])%2 else y[m] for m in Id], invsigma) for n in indices(Signd)])#y^(ee_sd)

            NYeesd=[NVd[np.all(Vd==Yeesd[n],axis=1)] for n in NSignd]#Vdに対応するものがあればその番号、なければemptyを返す
            exNSignd=NSignd[np.array([len(NYeesd[n])!=0 for n in NSignd])]#Vdに属するyeesdを与えるeps_の番号全体
            eta=np.array([np.all(1-xi(Yeesd[n],[(delta[isigma[m]]+eps[isigma[m]]+Signd[n][m])%2 for m in Id])) for n in exNSignd])#ブール反転
            trueNSignd=exNSignd[eta]
            if len(trueNSignd) > 0:
                ret.extend([[i,NYeesd[n][0],n] for n in trueNSignd])
    return ret


def classify(i):
    #rest=copy.deepcopy(NYE)
    rest=np.concatenate([[[j,k]for k in NSignd]for j in NVd])
    NYE0i=[]
    while len(rest)>0:
        ye=[rest[0][0],rest[0][1]]
        isom_pre2 = isom_sub(i, ye[0], ye[1])
        Isom=np.unique(isom_pre2,axis=0)
        NYE0i.append(Isom)
        rest=rest[[np.all(np.any(ye1!=Isom[:,1:],axis=1)) for ye1 in rest]]
    return NYE0i

d=3
N0=[[[0]],[[1]]]
#print(N0)
#d=2
N0.append([ [[2]],[[1,1]] ])

#d=3
N0.append([ [[3]],[[1,2]],[[1,1,1]] ])

#d=4
N0.append([[[4]],[[1,3],[2,2]],[[1,1,2]],[[1,1,1,1]]])

#d=5
N0.append([[[5]],[[1,4],[2,3]],[[1,1,3],[1,2,2]],[[1,1,1,2]],[[1,1,1,1,1]]])#X[i]はi個の自然数の和が5の組み合わせ

#d=6
N0.append([[[6]],[[1,5],[2,4],[3,3]],[[1,1,4],[1,2,3],[2,2,2]],[[1,1,1,3],[1,1,2,2]],[[1,1,1,1,2]],[[1,1,1,1,1,1]]])

#d=7
N0.append([[[7]],[[1,6],[2,5],[3,4]],[[1,1,5],[1,2,4],[1,3,3],[2,2,3]],[[1,1,1,4],[1,1,2,3],[1,2,2,2]],[[1,1,1,1,3],[1,1,1,2,2]],[[1,1,1,1,1,2]],[[1,1,1,1,1,1,1]]])

np.set_printoptions(threshold=np.inf)#numpy行列のprintを常に省略しない



Signd=Sign(d)
NSignd=np.arange(len(Signd))

S=Sym(d)
iS = inv(S)

e=np.arange(d)#[0,1,2,...,d-1]



Id=e
Vd=S.dot(e)
iVd=iS.dot(e)
NVd=np.arange(len(Vd))

Xrep=represent(N0[d],d)
iXrep=np.asarray([vinv(x) for x in Xrep])#X0の逆置換のベクトル表示を集める
X1=np.concatenate([NVd[np.all(Vd==x,axis=1)] for x in Xrep])
NX1=np.arange(len(X1))


##isomの代わりにマルチプロセッシングするためのモジュール
lVd=len(Vd)
lSignd=len(Signd)

#multiprocessing
n_process=1
n_thread=1

if __name__ == "__main__":
    #NYE0=copy.deepcopy(NX1)
    print(multiprocessing.cpu_count())
    #print(classify(0))
    #finish()
    #classifyをiを並列して処理する
    p = Pool(n_process)#プロセス数!!!!!!!!!
    NYE0 = p.map(classify, NX1)  # takeisom()にNX1の元を与えて並列演算
    p.close()
    #print(NYE0)
    t=time.time()
    dt=(t-t0)
    #print("#合計時間:",dt)
    #exit()
    NYE1=[np.array([[i,nye0[0][1],nye0[0][2]] for nye0 in NYE0[i]]) for i in NX1]
    YE1=[np.array([[Xrep[i],Vd[nye0[0][1]],Signd[nye0[0][2]]] for nye0 in NYE0[i]]) for i in NX1]
    lYE1=[len(NYE1[i]) for i in NX1]
    CNYE0=np.concatenate(NYE0)
    NCO=len(CNYE0)
    CNYE1=np.concatenate(NYE1)
    CNNYE1=np.array([i for i in indices(CNYE0)])
    CYE0=[[np.array([Xrep[a[0]],Vd[a[1]],Signd[a[2]]]) for a in c] for c in CNYE0]
    t = time.localtime()
    fname = str(t.tm_mon)+str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)
    with open('data_d={0}.txt'.format(d), 'w') as f:
        print('#d={0}'.format(d), file=f)
        print('import numpy as np'.format(d), file=f)
        print('NYE0=',NYE0, file=f)
        t=time.time()
        dt=(t-t0)
        print("#合計時間:",dt,file=f)
        finish()
