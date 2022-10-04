#coding: UTF-8

#inprementation of the algorithm in [G. Schmithüsen. An algorithm for ﬁnding the Veech group of an origami. Experiment. Math., 13(4):459–472, 2004]

#input
#L-shaped origami L(2,3)
x=[[1,2,3],[4]]
y=[[1,4],[2],[3]]


def strreverse(org_str):
	#new_str_list = list(reversed(org_str))
	new_str = ''.join(list(reversed(org_str)))
	return new_str
#文字列を左右逆向きにする

def Inverse(matrix):
	if (not (matrix=='I' or matrix=='J')):
		process1=matrix.translate(str.maketrans({'T':'U','U':'T','S':'a'+'S','J':'a'}))
		process2=strreverse(process1)
		if process2.count('a')%2==1:
			output='J'+process2.replace('a','')
		else:
			output=process2.replace('a','')
	else:
		output=matrix
	return output

#SL(2,Z)の行列を逆行列にする
#U=T^(-1), J=-I

def Pullback(object, matrix):
	if matrix=='T':
		output=object.translate(str.maketrans({'x':'x','y':'xy','z':'z','w':'wz'}))
	elif matrix=='U':
		output=object.translate(str.maketrans({'x':'x','y':'zy','z':'z','w':'wx'}))
	elif matrix=='S':
		output=object.translate(str.maketrans({'x':'y','y':'z','z':'w','w':'x'}))
	elif matrix=='J':
		output=object.translate(str.maketrans({'x':'z','y':'w','z':'x','w':'y'}))
	else: print('error')
	return output

#SL(2,Z)の元からβの引戻し(in Aut+(F2))を作る

def Back(path):
	return strreverse(Pullback(path, 'J'))
#パスを逆向きに辿る
#z=x^(-1), w=y^(-1)


def clean(path):
	Plist=list(path)
	i=0
	while i<len(Plist)-1:
		print(i, Plist[i], Back(Plist[i+1]),Plist)
		if Plist[i]==Back(Plist[i+1]):del Plist[i:i+2]
		#i,i+1番目の数は[i:i+2]のスライスに入る
		else:i=i+1
	return ''.join(Plist)
#行って戻る動きを解消する


def CalculatePath(path, matrix):
	lenM=len(matrix)
	process=[path]
	for i in range(lenM):
		process.append(process[i].translate(str.maketrans({'x':Pullback('x',matrix[i]),'y':Pullback('y',matrix[i]),'z':Pullback('z',matrix[i]),'w':Pullback('w',matrix[i])})))
		#process[i]にi番目の行列を作用させてprocess[i+1]を作る
		#range(lenM)={0,1,...,lenM-1}
	return process[lenM]
	#最後にできるのはi=lenM-1によるprocess[lenM]
#F2のパスをSL(2,Z)の元で動かしたものを求める(平行移動なし)

def getX(xcycles,ycycles,input):
	cycle=list(filter(lambda s:input in s, xcycles))[0]
	#イテレーターからそのまま要素取り出したい
	return cycle[(cycle.index(input)+1)%len(cycle)]
	#インデックス勝手にオーバーフローして0に戻ってほしい

def getY(xcycles,ycycles,input):
	cycle=list(filter(lambda s:input in s, ycycles))[0]
	return cycle[(cycle.index(input)+1)%len(cycle)]

def getZ(xcycles,ycycles,input):
	cycle=list(filter(lambda s:input in s, xcycles))[0]
	return cycle[(cycle.index(input)-1)%len(cycle)]

def getW(xcycles,ycycles,input):
	cycle=list(filter(lambda s:input in s, ycycles))[0]
	return cycle[(cycle.index(input)-1)%len(cycle)]

def getP(xcycles,ycycles,input):
	process0=input
	process1=getX(xcycles,ycycles,process0)
	process2=getY(xcycles,ycycles,process1)
	process3=getZ(xcycles,ycycles,process2)
	process4=getW(xcycles,ycycles,process3)
	return process4
#P=xyzw:puncture周りのループに対応する並び替えを取得

"""旧版
def getP(xcycles,ycycles,process0):
	cycle1=list(filter(lambda s:process0 in s, xcycles))[0]
	#イテレーターからそのまま要素取り出したい
	process1=cycle1[(cycle1.index(process0)+1)%len(cycle1)]
	#インデックス勝手にオーバーフローして0に戻ってほしい
	cycle2=list(filter(lambda s:process1 in s, ycycles))[0]
	process2=cycle2[(cycle2.index(process1)+1)%len(cycle2)]
	cycle3=list(filter(lambda s:process2 in s, xcycles))[0]
	process3=cycle3[(cycle3.index(process2)-1)%len(cycle3)]
	cycle4=list(filter(lambda s:process3 in s, ycycles))[0]
	process4=cycle4[(cycle4.index(process3)-1)%len(cycle4)]
	return process4
"""

def Origami(xcycles,ycycles):
	d=sum([len(x) for x in xcycles])
	x=xcycles
	y=ycycles
	cells=range(1,d+1)
	#RepHを作る
	#RepHのインデックスはセル番号と対応させる
	RepH=['']*d
	GenH=[]
	checked=[1]#RepHラベル済みセル番号の集合
	for c in cells:
		#x,y方向の移動先を見る
		if getX(x,y,c) in checked:
			GenH.append(RepH[c-1]+'x'+Back(RepH[getX(x,y,c)-1]))
			#自明なものを含め、重複して得られる？
			#セルcの代表元はRepH[c-1]
		else:
			RepH[getX(x,y,c)-1]=RepH[c-1]+'x'
			checked.append(getX(x,y,c))
		if getY(x,y,c) in checked:
			GenH.append(RepH[c-1]+'y'+Back(RepH[getY(x,y,c)-1]))
		else:
			RepH[getY(x,y,c)-1]=RepH[c-1]+'y'
			checked.append(getY(x,y,c))
	return (RepH,GenH)
	#RepHの完全な表現とGenHの十分な表現を得た
	#停止性:xy方向だけで全部のセルにたどり着けることの証明？
	#GenHも完全なリストなのでは？
#	M=len(x)
#	N=len(y)
#	checked=[]
#	#Pcyclesを作る
#	Pcycles=[]
#	j=0#cycle番号に対応
#	for c in cells:
#		if c in checked:continue
#		Pcycles.append([c])
#		Pc=getP(x,y,c)
#		while Pc!=c:
#			Pcycles[j].append(Pc)
#			Pc=getP(xcycles,ycycles,Pc)
#		checked=list(set(checked)|set(Pcycles[j]))
#		j=j+1
#	#GenHの完全なリストを構成するのは必要？





def LoopCheck(matrix, GenH, xcycles, ycycles):
	d=sum([len(x) for x in xcycles])
	N=len(GenH)
	x=xcycles
	y=ycycles
	cells=range(1,d+1)
	result=False
	for i in cells:#各セルからのスタートで調べる
		help=True
		for j in range(N):#すべての基底について調べる
			root=CalculatePath(GenH[j],matrix)
			#j番目の基底を変換してrootとした
			now=i
			for k in range(len(root)):
				if root[k]=='x':now=getX(x,y,now)
				elif root[k]=='y':now=getY(x,y,now)
				elif root[k]=='z':now=getZ(x,y,now)
				elif root[k]=='w':now=getW(x,y,now)
			#rootに沿ってセルiから辿った
			if now!=i:help=False
		if help:result=True
	return result
	
def MainAlgorithm(GenH,xcycles,ycycles):
	d=sum([len(x) for x in xcycles])
	N=len(GenH)
	x=xcycles
	y=ycycles
	H=GenH
	Gen=[]
	Rep=['']
	n=0
	while n < len(Rep):
		A=Rep[n]
		for B in [A+'T', A+'S']:
			help=False
			for m in range(len(Rep)):
				C=B+Inverse(Rep[m])
				if LoopCheck(C,H,x,y):
					help=True
					Gen.append(C)
			if not(help):Rep.append(B)
		n=n+1
	Rep[0]='I'
	return (Rep, Gen)
		
		
		
	


d=sum([len(a) for a in x])
#x,y,dを指定
RepH=Origami(x,y)[0]
GenH=Origami(x,y)[1]

result=MainAlgorithm(GenH,x,y)
print('x=', x)
print('y=', y)
print('のとき、')
print('Rep=', result[0])
print('Gen=', result[1])



"""
def Origami(GenH)
	size=len(GenH)
	step=[x,y,z,w]
	RepH=[1]
	RepHold=[]
	while RepH==RepHold:
		RepHold=RepH
		for c in RepH
			for s in step
				help=true
				#csに組み合わせてGenHに入るRepHの元がないか判定
				for d in RepH
					if GenH.count(clean(c+s+Back(d)))+GenH.count(clean(d+Back(c+s)))!=0:help=false
				if help:RepH.append(c+s)
		
#GenHからRepHとxcycles,ycyclesを取り出すのは大変そう
"""
	
