import time
t1=time.time()
t0=time.time()

import itertools as it
import numpy as np
import math
import pickle
import copy
from functools import reduce

def lcm_base(x, y):
    return (x * y) // math.gcd(x, y)

def lcm(*numbers):
    return reduce(lcm_base, numbers, 1)

def lcm_list(numbers):
    return reduce(lcm_base, numbers, 1)

def indices(A):
    return range(len(A))

def Cdelta(a,b):
    D=1 if a==b else 0
    return D
#

def M(x):#x(ベクトル表示)の基本行列表示の逆行列(転置)
    l=len(x)
    return np.asarray([[Cdelta(j,x[i])for i in range(l)] for j in range(l)])

def inv(A):#行列式1の行列の逆行列(転置行列)
    l=len(A)
    return np.asarray([[A[i][j] for i in range(l)] for j in range(l)])



def iM(x):
    return inv(M(x))

def tttime():
    t=time.time()
    dt=(t-t0)
    print("経過時間:",dt)

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
    S=np.asarray([[[Cdelta(i,s[j])for i in range(d)] for j in range(d)] for s in list(it.permutations(list(range(d))))])
    return S



test=inv(Sym(3)[3]).dot([0,1,2])
test2=inv(Sym(3)[3]).dot([3,4,5])
print(cycle([1,2,0]))
print(cycle([1,2,0,4,3]))
print(icycle(cycle([1,2,0,4,3])))
print((1==5)==False)
#rest=np.asarray([0,1,2,3])
#print(rest==2)
#rest=rest[rest!=2]
#print(rest)
#finish()
#print(Sym(3)[3])#original matrix
#print(test)#vector representation
#print(M(test))#re-matricize(identical)
#print(test2)#pullback

def isom_bu(x,y):
    d=len(x)
    S=Sym(d)
    l=len(S)
    Rl=range(l)
    Rd=range(d)
    e=np.arange(1,d+1)
    E1=S.dot(e)
    E2=np.asarray([iperm(E1[i]) for i in Rl])
    X1=np.asarray([[E1[k][x[i]-1] for i in Rd] for k in Rl])
    X2=np.asarray([E2[k].dot(X1[k]) for k in Rl])
    Y1=np.asarray([[E1[k][y[i]-1] for i in Rd] for k in Rl])
    Y2=np.asarray([E2[k].dot(Y1[k]) for k in Rl])
    Y3=Y2[np.all(X2==x, axis=1)]
    #print(Y3)
    return Y3

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

d=3
N0=[[[0]],[[1]]]
print(N0)
#d=2
N0.append(np.asarray([ [[2]],[[1,1]] ]))

#d=3
N0.append(np.asarray([ [[3]],[[1,2]],[[1,1,1]] ]))

#d=4
N0.append(np.asarray([[[4]],[[1,3],[2,2]],[[1,1,2]],[[1,1,1,1]]]))

#d=5
N0.append(np.asarray([[[5]],[[1,4],[2,3]],[[1,1,3],[1,2,2]],[[1,1,1,2]],[[1,1,1,1,1]]]))#X[i]はi個の自然数の和が5の組み合わせ

#d=6
N0.append(np.asarray([[[6]],[[1,5],[2,4],[3,3]],[[1,1,4],[1,2,3],[2,2,2]],[[1,1,1,3],[1,1,2,2]],[[1,1,1,1,2]],[[1,1,1,1,1,1]]]))

#d=7
N0.append(np.asarray([[[7]],[[1,6],[2,5],[3,4]],[[1,1,5],[1,2,4],[1,3,3],[2,2,3]],[[1,1,1,4],[1,1,2,3],[1,2,2,2]],[[1,1,1,1,3],[1,1,1,2,2]],[[1,1,1,1,1,2]],[[1,1,1,1,1,1,1]]]))

np.set_printoptions(threshold=np.inf)#numpy行列のprintを常に省略しない


#def Sym(d):
#    S=np.asarray([[[Cdelta(i,s[j])for i in range(d)] for j in range(d)] for s in list(it.permutations(list(range(d))))])
#    return S

def Sign(d):
    return np.asarray([[(t//(2**s))%2 for s in range(d)] for t in range(2**d)])
Signd=Sign(d)
NSignd=np.arange(len(Signd))

S=Sym(d)
iS=np.asarray([inv(A) for A in S])

e=np.arange(d)#[0,1,2,...,d-1]

def vinv(v):
    return M(v).dot(e)

Id=e
Vd=S.dot(e)
iVd=iS.dot(e)
NVd=np.arange(len(Vd))

Xrep=represent(N0[d],d)
iXrep=np.asarray([vinv(x) for x in Xrep])#X0の逆置換のベクトル表示を集める
#X1=np.arange(len(Xrep))#numbering Xrep
X1=np.concatenate([NVd[np.all(Vd==x,axis=1)] for x in Xrep])
NX1=np.arange(len(X1))

def conjugate(v,x):
#    return iM(v).dot([vinv(v)[x[j]] for j in range(d)])#逆にした
    return ([v[ x[vinv(v)[j]] ] for j in range(d)])

#vvv=[1,2,0]
#print(vinv(vvv))
#print(conjugate(vvv,vvv))
#finish()

Xclass=[[conjugate(v,x) for v in Vd] for x in Xrep]


"""
#X0=[np.concatenate([NVd[np.all(Vd==iM(v).dot([v[x[j]] for j in range(d)]),axis=1)] for v in Vd]) for x in Xrep]
X0=[np.concatenate([NVd[np.all(Vd==conjugate(v,x),axis=1)] for v in Vd]) for x in Xrep]#X0[i][j]はX1[i]のVd[j]でのconjugateに対応
print(Xrep, NVd, X0)
def searchX0(x):#xがX1の何番目に属するかの出力
    return NX1[np.any(X0==NVd[np.all(Vd==x,axis=1)],axis=1)]

NE=[[[[i,j,k]for k in NSignd]for j in NVd] for i in NX1]

"""

"""#signを符号列として参照するバージョンだが、不要か
def ytosign(j,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return [Vd[j][n] if sign[n] == 0 else iVd[j][n] for n in Id]

def xtosign(i,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return [Xrep[i][n] if sign[n] == 0 else iXrep[i][n] for n in Id]

def vtosign(v,sign):#ベクトルを参照するバージョン
    return v if sign == 0 else vinv(v)
"""

#signは符号列ではなく符号値として参照する
def ytosign(j,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return Vd[j] if sign == 1 else iVd[j]

def xtosign(i,sign):#sign=0,falseは正、sign=1,trueは負として処理
    return Xrep[i] if sign == 1 else iXrep[i]

def vtosign(v,sign):#ベクトルを参照するバージョン
    return v if sign == 1 else vinv(v)
"""
#x,y,eps -> boldx,boldy in canonical double
def bx(x,y,eps,pm,i):
    return [pm, vtosign(x,pm)[i]]
def ibx(x,y,eps,pm,i):
    return [pm, vtosign(x,1-pm)[i]]

def by(x,y,eps,pm,i):
    return [(pm+eps[i]+eps[vtosign(y,eps[i]!=pm)[i]])%2,vtosign(y,eps[i]!=pm)[i]]
def iby(x,y,eps,pm,i):
    return [(pm+eps[i]+eps[vtosign(y,eps[i]==pm)[i]])%2,vtosign(y,eps[i]==pm)[i]]
"""
#x,y,eps -> boldx,boldy in canonical double
#(0,0~d-1)->(0~d-1) (1,0~d-1)->(d~2d-1) version
def bx(x,y,eps,j):
    pm=j//d
    i=j%d
    bx=[pm, vtosign(x,1-pm)[i]]
    return d*bx[0]+bx[1]

def ibx(x,y,eps,j):
    pm=j//d
    i=j%d
    ibx=[pm, vtosign(x,pm)[i]]
    return d*ibx[0]+ibx[1]

def by(x,y,eps,j):
    pm=j//d
    i=j%d
    by=[(pm+eps[i]+eps[vtosign(y,eps[i]==pm)[i]])%2,vtosign(y,eps[i]==pm)[i]]
    return d*by[0]+by[1]

def iby(x,y,eps,j):
    pm=j//d
    i=j%d
    iby=[(pm+eps[i]+eps[vtosign(y,eps[i]!=pm)[i]])%2,vtosign(y,eps[i]!=pm)[i]]
    return d*iby[0]+iby[1]

print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@")

def isom(i,j,k):
    x=Xrep[i]
    eps=Signd[k]
    #Stabx=Vd[[np.all(conjugate(v,x)==x) for v in Vd]]
    #NStabx=NVd[[np.all(conjugate(v,x)==x) for v in Vd]]
    isom=[]
    for ns in NVd:#sigma=Vd[ns]
        sigma=Vd[ns]
        isigma=iVd[ns]
        #print("sigma=",sigma)
        for nd in NSignd:#delta=Signd[nd]
            delta=Signd[nd]
            #print("nd=",nd,"delta=",delta)
            if np.any(delta!=np.array([delta[x[m]] for m in Id])):
                #print("error1")
                continue
            if np.any(conjugate(sigma,x)!=[xtosign(i,not delta[m])[m] for m in Id]):
                #print(x,sigma,conjugate(sigma,x))
                #print([xtosign(i,not delta[m])[m] for m in Id])
                #print("error2")
                continue
            yde=[ytosign( j,delta[isigma[m]]==eps[isigma[m]] )[isigma[m]] for m in Id]#符号積は==
            yde_=[ytosign( j,delta[isigma[m]]!=eps[isigma[m]] )[isigma[m]] for m in Id]
            mu=[sigma[yde[m]] for m in Id]
            nu=[sigma[yde_[m]] for m in Id]
            eta=np.array([eps[isigma[m]]==eps[yde[m]]for m in Id])
            eta_=np.array([ [eps_[m]==(delta[isigma[m]]+delta[yde[m]]+eps_[mu[m]])%2 for m in Id] for eps_ in Signd])
            """
            for m in Id:
                #mu[m]=conjugate(sigma, ytosign(j,eps[m]))[m]#なんかちがう:ytosignがうまくconjugateさせられてない
                #nu[m]=conjugate(sigma, ytosign(j,1-eps[m]))[m]#なんかちがう
                mu[m]=conjugate(sigma, ytosign(j,delta[isigma[m]]!=eps[isigma[m]]))[m]#
                nu[m]=conjugate(sigma, ytosign(j,delta[isigma[m]]==eps[isigma[m]]))[m]#TrueFalseの値処理を逆にしたので反転してる
                eta[m]=( eps[isigma[m]]==eps[ ytosign(j,delta[isigma[m]]!=eps[isigma[m]])[isigma[m]] ] )
            """
            #print(np.all(eta_==eta,axis=1))
            #print("Stab check OK")
            #print("delta=",delta)
            #print("sigma=",sigma)
            #print("yde=",yde)
            #print("eta=",eta)
            #print("eta'=",eta_)
            #print([ [[delta[isigma[m]],delta[yde[m]],eps_[mu[m]]] for m in Id] for eps_ in Signd])
            #print(np.all(eta_==eta,axis=1))
            Eps_=Signd[np.all(eta_==eta,axis=1)]
            NEps_=NSignd[np.all(eta_==eta,axis=1)]
            #print("i,j,k=",i,j,k)
            #print("x,y,e=",Xrep[i],Vd[j],Signd[k])
            """
            if np.all(Xrep[i]==[1,2,0]) and np.all(Vd[j]==[1,2,0]):
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@")
            """
            #print("mu,nu=",mu,nu)
            #print(Signd)
            Y_=[[(mu[m] if eps_[m]==0 else nu[m]) for m in Id]for eps_ in Eps_]
            #print("Eps'=",Eps_)
            #print("Y_=",Y_)
            #print("finish")
            NY_=[NVd[np.all(Vd==y_,axis=1)] for y_ in Y_]#通し番号！
            exY_=np.array([len(NY_[m])!=0 for m in indices(Y_)])
            if len(exY_)==0:continue
            NY_2=np.array(NY_)[exY_]
            Y_2=np.array(Y_)[exY_]
            Eps_2=Eps_[exY_]
            NEps_2=NEps_[exY_]
            #print(NY_)
            isom.append([[i,NY_2[l][0],NEps_2[l]]for l in indices(Eps_2)])
    return np.unique(np.concatenate(np.array(isom)),axis=0)#出力は[ny_, neps_]=[j,k]が並ぶ形

#input=(0,1,0)
#output=isom(input[0],input[1],input[2])

#print(Vd)
#print("input=",Vd[input[0]],[Vd[input[1]],Signd[input[2]]])
#print("output=",Xrep[input[0]],[np.array([Vd[output[t][0]],Signd[output[t][1]]]) for t in indices(output)],len(output))
#print(NSignd)
#finish()


NYE=np.concatenate([[[j,k]for k in NSignd]for j in NVd])
YE=np.array([[ Vd[nye[0]],Signd[nye[1]] ] for nye in NYE])



def enumerate():
    NYE0=[]
    for i in NX1:
        #rest=copy.deepcopy(NYE)
        rest=np.concatenate([[[i,j,k]for k in NSignd]for j in NVd])
        NYE0i=[]
        while len(rest)>0:
            ye=[rest[0][1],rest[0][2]]
            #print(i,ye)
            Isom=isom(i,ye[0],ye[1])
            NYE0i.append(Isom)
            #print(i,Isom,rest)
            rest=rest[[np.all(np.any(ye1!=Isom,axis=1)) for ye1 in rest]]
            print("rep=",[i,ye[0],ye[1]],", 残り",len(rest),"パターン")
        NYE0.append(NYE0i)
        t=time.time()
        dt=(t-t1)
        print("i=",i,"終了, 経過時間:",dt)
    return NYE0

#NYE0にxの番号も格納するようにして全てのnyeの参照を[0],[1]→[1],[2]に変えたい

#https://www.yoheim.net/blog.php?q=20171004
import importlib
from importlib import import_module
data = import_module( 'data_d={}'.format(d))
NYE0=data.NYE0


#NYE0=enumerate()#一番時間かかる操作
NYE1=[np.array([[i,nye0[0][1],nye0[0][2]] for nye0 in NYE0[i]]) for i in NX1]

YE1=[np.array([[Xrep[i],Vd[nye0[0][1]],Signd[nye0[0][2]]] for nye0 in NYE0[i]]) for i in NX1]

#NNYE1=[[[i,j] for j in indices(NYE1[i])] for i in NX1]#[x,ye]の番号集合

def restore(nxye):
    return [cycle(Xrep[nxye[0]]),cycle(Vd[nxye[1]]),Signd[nxye[2]]]

lYE1=[len(NYE1[i]) for i in NX1]



CNYE0=np.concatenate(NYE0)
NCO=len(CNYE0)
CNYE1=np.concatenate(NYE1)
CNNYE1=np.array([i for i in indices(CNYE0)])

CYE0=[[np.array([Xrep[a[0]],Vd[a[1]],Signd[a[2]]]) for a in c] for c in CNYE0]
CYE1=[c[0] for c in CYE0]
#print("NYE0=",NYE0)

#print(Xrep)


def Orbit(x,y):#軌道分解をする
    if len(x)!=len(y):print("error:Orbitlength")
    n=len(x)
    cx=cycle(x)
    Ncx=np.array(indices(cx))
    cy=cycle(y)
    Ncy=np.array(indices(cy))
    rest=np.array(range(n))
    decomp=[]
    while len(rest)!=0:
        i=rest[0]
        orbi=[i]#iが属する軌道
        resti=[i]
        donei=[]#orbi上で処理した添え字
        while len(resti)!=0:
            j=resti[0]
            cxj=cx[Ncx[([np.any(np.array(cx[k])==j) for k in indices(cx)])][0]]#jを含むxのサイクル
            cyj=cy[Ncy[([np.any(np.array(cy[k])==j) for k in indices(cy)])][0]]#jを含むyのサイクル
            orbi=np.unique(np.concatenate([orbi,cxj,cyj]))
            #print(cx,cy,orbi,rest)
            resti=copy.deepcopy(orbi)
            donei.append(j)
            for l in donei:
                resti=resti[resti!=l]
            for l in orbi:
                rest=rest[rest!=l]
        decomp.append(orbi)
        #print(rest)
    return decomp

print("start")
def PermT():
    permT=[[0 for j in NYE1[i]]for i in NX1]
    for i in NX1:
        x=Xrep[i]
        ix=iXrep[i]
        for j in indices(NYE1[i]):
            nye1=NYE1[i][j]#=[i,j,k]の形, i=nye[0]
            y=Vd[nye1[1]]
            eps=Signd[nye1[2]]
            #print("i,j=",i,j)
            #print("x,y,eps=",cycle(x),cycle(y),eps)
            cyT=[]#サイクル表示を作る
            epsT=[0 for i in Id]
            rest=copy.deepcopy(Id)
            while len(rest)!=0:
                a1=rest[0]
                a2=rest[0]
                a0=rest[0]
                cyT1=[a0]
                #epsT1=[0]
                eps_1=0
                while True:#a0始動のサイクルを一つ作る:kはこのサイクル内の順番
                    #print("a1=",a1)
                    #print("eps_1=",eps_1)
                    rest=rest[rest!=a2]#remove a1 from rest
                    a2_=vtosign(y,not eps_1!=eps[vtosign(x,not eps_1)[a1]])[vtosign(x,not eps_1)[a1]]
                    #print(eps_1!=eps[vtosign(x,not eps_1)[a1]])
                    eps_2=((eps_1!=eps[vtosign(x,not eps_1)[a1]])!=eps[a2_])#0=+,1=-としたとき符号の積は!=でかける
                    if eps_2==0:a2=a2_#a2はconjugateした出力としてのみ用意
                    else:a2=ix[a2_]
                    #print("a2_=",a2_,"a2=",a2,a0)
                    if a2==a0:break
                    #cyT1[len(cyT1)]=a2
                    cyT1=cyT1+[a2]
                    epsT[a2]=eps_2
                    eps_1=eps_2
                    #print("eps_2=",eps_2)
                    #print(epsT)
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    a1=a2_
                cyT=cyT+[cyT1]
            epsT=np.array(epsT).astype(np.int)
            #print(cyT,epsT)
            #print("")
            yT=icycle(cyT,d)
            permT[i][j]=np.concatenate([[i],NYE[np.all(np.all(YE==[yT,epsT],axis=2),axis=1)][0]])
            #print(NYE[np.all(np.all(YE==[yT,epsT],axis=2),axis=1)][0],np.concatenate([[i],NYE[np.all(np.all(YE==[yT,epsT],axis=2),axis=1)][0]]))
            #finish()
            #if i==1 and j==13:finish()
    return permT
#print("PermT=",PermT())


#print("BEGIN:S")
debugflagM=False
def PermS():
    debugflag=False
    permS=[[0 for j in NYE1[i]]for i in NX1]
    for i in NX1:
        x=Xrep[i]
        ix=iXrep[i]
        for j in indices(NYE1[i]):
            nye1=NYE1[i][j]#=[i,j,k]の形, i=nye[0]
            y=Vd[nye1[1]]
            eps=Signd[nye1[2]]
            #print("x,y,eps=",x,y,eps)


            if debugflagM:
                #if i==0 and nye1[1]==21 and nye1[2]==4:debugflag=True
                if i==4 and nye1[1]==311 and nye1[2]==36:debugflag=True
            if debugflag:print(x,y,eps)


            #deltaの導出にミス有
            cxS=[]#サイクル表示を作る
            deltaS=[0 for i in Id]
            rest=copy.deepcopy(Id)
            while len(rest)!=0:#取りつくしでサイクル表示を作るアルゴリズム
                a1=rest[0]
                a2=rest[0]
                a0=rest[0]
                cxS1=[a0]
                delta1=0
                while True:#a0始動のサイクルを一つ作る
                    #print(a0)
                    rest=rest[rest!=a2]#remove a2 from rest
                    a2=vtosign(y,not eps[a1]!=delta1)[a1]
                    delta2=(delta1!=eps[a1])!=eps[a2]#0=+,1=-としたとき符号の積は!=でかける
                    if a2==a0:break
                    cxS1=cxS1+[a2]
                    deltaS[a2]=delta2
                    delta1=delta2
                    a1=a2
                cxS=cxS+[cxS1]
            #print("cxS,deltaS=",cxS,deltaS)
            xS_=icycle(cxS,d)

            cyS=[]#サイクル表示を作る
            epsS_=[0 for i in Id]
            rest=copy.deepcopy(Id)
            while len(rest)!=0:#取りつくしでサイクル表示を作るアルゴリズム
                b1=rest[0]
                b2=rest[0]
                b0=rest[0]
                cyS1=[b0]
                epsS_1=0
                while True:#b0始動のサイクルを一つ作る
                    rest=rest[rest!=b2]#remove b2 from rest
                    b2=vtosign(x,not not epsS_1!=deltaS[b1])[b1]
                    #b2=vtosign(x,not not deltaS[b1])[b1]
                    epsS_2=(epsS_1!=deltaS[b1])!=deltaS[b2]#0=+,1=-としたとき符号の積は!=でかける
                    #epsS_2=(deltaS[b2]!=epsS_1)
                    if debugflag:print(b2,epsS_2)
                    if b2==b0:
                        #print("break")
                        break
                    #print(b0,b2)
                    cyS1=cyS1+[b2]
                    epsS_[b2]=epsS_2#
                    epsS_1=epsS_2
                    b1=b2
                cyS=cyS+[cyS1]
            epsS_=np.array(epsS_).astype(np.int)##########################################
            yS_=icycle(cyS,d)
            """
            if i==0 and np.all(y==Vd[10]):
                #print("x,y,eps=",x,y,eps)
                print("yS_=",yS_)
                #print("")
            """
            #xSをRepXに入れるようにconjugateする
            Xnum=NX1[np.any(np.all(Xclass==np.array(xS_),axis=2),axis=1)][0]#xS belongs to Xrep[Xnum]
            conj=iVd[np.all(Xclass==np.array(xS_),axis=2)[Xnum]][0]#conjugator#Xclass={v*xrep v in Vd}だから、conj*xS_=xSとするためにはiVdから引っ張ってこないといけなかった
            xS=Xrep[Xnum]
            if np.any(xS!=conjugate(conj,xS_)):
                print("conjugate error")
                finish()
            yS=conjugate(conj,yS_)#ayasi
            epsS=[epsS_[vinv(conj)[i]] for i in Id]
            #print("")
            #print("")
            #print(i,j)
            permS[i][j]=np.concatenate([[Xnum],NYE[np.all(np.all(YE==[yS,epsS],axis=2),axis=1)][0]])
            #print(NYE[np.all(np.all(YE==[yT,epsT],axis=2),axis=1)][0],np.concatenate([[i],NYE[np.all(np.all(YE==[yT,epsT],axis=2),axis=1)][0]]))
    return permS


permT=np.concatenate(PermT())
print("PermT:done")
tttime()
permS=np.concatenate(PermS())
print("PermS:done")
tttime()
#行き先の[Nx,Ny,Neps]を返す
permT1=[CNNYE1[([np.any(np.all(CNYE0[j]==permT[i],axis=1)) for j in indices(CNYE0)])][0] for i in CNNYE1]
permS1=[CNNYE1[([np.any(np.all(CNYE0[j]==permS[i],axis=1)) for j in indices(CNYE0)])][0] for i in CNNYE1]
permTS1=[permT1[permS1[i]] for i in CNNYE1]

#print(CYE1[2292])
#print(CYE1[3458])
#print(CYE1[2272])
print("")
#print(CYE1[2293])
#print(CYE1[3457])
#print(CYE1[2273])
#print(permS1)
errorflag=False
def errorcycle(v):
    for i in indices(v):
        if v[v[i]]!=i:
            errorflag=True
            print("error. i,v[i],v2[i],v3[i],v4[i]=",i,v[i],v[v[i]],v[v[v[i]]],v[v[v[v[i]]]])
            print("i")
            print(CNYE1[i][0],CNYE1[i][1],CNYE1[i][2])
            print(cycle(CYE1[i][0]),cycle(CYE1[i][1]),CYE1[i][2])
            print("v[i]")
            print(CNYE1[v[i]][0],CNYE1[v[i]][1],CNYE1[v[i]][2])
            print(cycle(CYE1[v[i]][0]),cycle(CYE1[v[i]][1]),CYE1[v[i]][2])
            print("v2[i]")
            print(CNYE1[v[v[i]]][0],CNYE1[v[v[i]]][1],CNYE1[v[v[i]]][2])
            print(cycle(CYE1[v[v[i]]][0]),cycle(CYE1[v[v[i]]][1]),CYE1[v[v[i]]][2])
            print("v3[i]")
            print(CNYE1[v[v[v[i]]]][0],CNYE1[v[v[v[i]]]][1],CNYE1[v[v[v[i]]]][2])
            print(cycle(CYE1[v[v[v[i]]]][0]),cycle(CYE1[v[v[v[i]]]][1]),CYE1[v[v[v[i]]]][2])
            print(" ")

errorcycle(permS1)
print("errorcycle:done")
if errorflag:finish()

#行き先のCNY1(通し番号)を返す
CpermT=cycle(permT1)
CpermS=cycle(permS1)
CpermTS=cycle(permTS1)

Orb=Orbit(permT1,permS1)

def output():
    with open('result2_d={0}.txt'.format(d), 'w') as f:
        print(" ")
        print("output")
        print("  d =",d)
        print("#d =",d, file=f)
        print(" ")
        Num=1
        debugnum=0
        for orb in Orb:#multiprocessできそう
            rep=orb[0]
            debugnum=debugnum+1
            #print(np.any(np.any(CNYE0[rep]==[d,d,0],axis=1)))
            if np.any(np.any(CNYE0[rep]==[-1,-1,0],axis=1)) or np.any(np.any(CNYE0[rep]==[-1,-1,len(Signd)-1],axis=1)):Abelian=True
            else:Abelian=False
            Nxye=CNYE1[rep]
            x=Xrep[Nxye[0]]
            y=Vd[Nxye[1]]
            eps=Signd[Nxye[2]]
            decomp=Orbit(x,y)
            if len(decomp)!=1:
                #print("disconnected")
                continue
            """
            z=[vinv(y)[vinv(x)[y[x[i]]]] for i in Id]
            Nv=len(cycle(z))
            """
            if Abelian:
                z=[vinv(y)[vinv(x)[y[x[i]]]] for i in Id]
                vl=np.sort([4*len(c) for c in cycle(z)])#valency list
                Nv=len(vl)#number of vertices
            else:
                bz=[iby(x,y,eps,ibx(x,y,eps,by(x,y,eps,bx(x,y,eps,j)))) for j in range(2*d)]
                cbz=cycle(bz)
                #print("Num=",Num)
                #print(cbz)
                vld=[]#valency list of double-paired vertices
                vlr=[]#valency list of ramified vertices
                for c in cbz:
                    #print(c)
                    bxby0=by(x,y,eps,bx(x,y,eps,c[0]))
                    #print(bxby0)
                    cbz0=[c1 for c1 in cbz if np.any(np.array(c1)==c[0])][0]
                    #print(cbz0)
                    if np.any(cbz0==bxby0%d+(1-bxby0//d)*d):#ramified vertex#np.any(cbz0==bxby0) or
                        #print("a")
                        vlr.append(len(c)*2)
                    else:#double-paired vertex
                        vld.append(len(c)*4)
                vldr=[np.sort(vld)[2*i] for i in range(len(vld)//2)]
                vl=np.sort(vlr+vldr)#valency list
                #print(vl)
                Nv=len(vl)#number of vertices
            CT=[c for c in CpermT if np.any(orb==c[0])]
            CS=[c for c in CpermS if np.any(orb==c[0])]
            CTS=[c for c in CpermTS if np.any(orb==c[0])]
            genus=1-(len(orb)-(3*len(orb)-len([c for c in CS if len(c)==1]))/2+len(CT)+len(CTS))/2
            WL=lcm(*[len(c) for c in CT])
            print("  Component No.",Num, file=f)
            print("  representatives: ",orb, file=f)
            print("  index of VG =",len(orb), file=f)
            print("  base: (x,y,eps) = (",cycle(x),cycle(y),eps,")", file=f)
            #print("  surface type= (",int((d-Nv)/2+1),",",Nv,")", file=f)
            #print("  valency list= ",vl , file=f)
            #print("  Abelian:",Abelian, file=f)
            print("  stratum:","A_" if Abelian else "Q_",int((d-Nv)/2+1),[int(v/2-2) for v in vl],   file=f)
            #print("  T =",CT, file=f)
            print("  widths list of T =",[len(c) for c in CT],len(CT), file=f)
            #print("  S =",CS, file=f)
            print("  widths list of S =",[len(c) for c in CS],len(CS), file=f)
            #print("  TS =",CTS, file=f)
            print("  widths list of TS =",[len(c) for c in CTS],len(CTS), file=f)
            print("  genus =",math.floor(genus), file=f)
            print("  Wolfarht level =",WL, file=f)
            """
            for i in orb:
                print("    representative No.",i, file=f)
                Nxyei=CNYE1[i]
                xi=Xrep[Nxyei[0]]
                yi=Vd[Nxyei[1]]
                epsi=Signd[Nxyei[2]]
                print("    (x,y,eps) = (",cycle(xi),cycle(yi),epsi,")", file=f)
            """
            print(" ", file=f)
            print(" ", file=f)
            print(" ", file=f)
            Num=Num+1
            #if debugnum==5:finish()
    print("end")


print("Output")
output()


finish()
count=0
for c in CYE0:
    print("class No.",count)
    for a in c:
        print([cycle(a[0]),cycle(a[1]),a[2]])
    print(" ")
    if count==4:finish()
    count=count+1
finish()
finish()
















t2=time.time()
dt=(t2-t1)
print("経過時間:",dt)

#with open("resultO6.py", "w") as f:
#    print("time=",dt,"permT=",permT, file=f)
#埼玉大　竹内先生
