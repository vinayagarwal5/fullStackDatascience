import sys
sys.setrecursionlimit(100)
print(sys.getrecursionlimit())

i=0

def wish():
    global i
    i +=1
    print('hello',i)
    if i < 5:
        wish()
wish()