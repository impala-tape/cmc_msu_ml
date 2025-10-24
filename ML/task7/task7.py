def find_modified_max_argmax(L,f):
    a=[]
    for x in L:
        if type(x)is int:a.append(f(x))
    if not a:return()
    m=max(a)
    return m,a.index(m)