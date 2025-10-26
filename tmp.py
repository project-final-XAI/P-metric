change(A,K,C)
    for i = 1 to k
        do x[i] = 0

    i=k
    while A>0
        do if A>pow(c,i)
        then 
            x[i]=x[i]+1
            A=A-pow(c,i)
        else
            i=i-1
        
    return x