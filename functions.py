import numpy as np
import matplotlib.pyplot as plt
import time
import random
import cvxpy as cp
from scipy.optimize import direct
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

solver="MOSEK"


def linear(x,param):
    A=param[0]
    b=param[1]
    return cp.max(A@x-b)
    
def quadratic(x,param):
    Q=param[0]
    q=param[1]
    return cp.quad_form(x,Q)-q

def generic_fun(x,param=[]):
    fun=param[0]
    fun_param=param[1]
    return fun(x,fun_param)


def vf(xk,g,Y):
    numb_iters=100
    def gneg(y,x):
        return -g(x,y)
    res=direct(gneg,Y,args=(xk,), maxiter=numb_iters)
    return -res.fun,res.x


def vf_concave(xk,yk,gcp,Y):
    y = cp.Variable(yk.shape)
    cons_y=[]
    for i in range(len(Y)):
        gi=Y[i][0]
        param_gi=Y[i][1]
        cons_y=cons_y+[gi(y,param_gi)<=0]
    prob = cp.Problem(cp.Maximize(gcp(xk,y)),cons_y)
    prob.solve(solver)
    return prob.value,y.value

def vf_maximum_concave(xk,yk,gcps,Y):
    value=-100000
    ymax=yk
    for i in range(len(gcps(xk,yk))):
        def gcpi(xk,yk):
            return gcps(xk,yk)[i]
        v,yv=vf_concave(xk,yk,gcpi,Y)
        if v>=value:
            value=v
            ymax=yv
    return value,ymax
        
    
def vf_disc(xk,g,Yk):
    maxind=np.argmax([g(yk,x) for yk in Yk])
    return g(Yk[maxind],x),Yk[maxind]

def compute_min_disc(xk,fcp,gcp,X,Yk,rho,J):
    n=np.size(xk)
    x = cp.Variable(n)
    eta=cp.Variable(1)
    cons_eta=[eta>=0]
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    for j in J:
        cons_eta=cons_eta+[eta>=gcp(x,Yk[j])]
    prob = cp.Problem(cp.Minimize(fcp(x)+rho*eta),cons_x+cons_eta)
    prob.solve(solver)
    return prob.value,x.value,eta.value

def compute_min_disc1(xk,fcp,gcp,X,Yk,rho,J):
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    q=cp.hstack([gcp(x,Yk[j]) for j in J])
    prob = cp.Problem(cp.Minimize(fcp(x)+rho*cp.pos(cp.max(q))),cons_x)
    prob.solve(solver)
    return prob.value,x.value

def compute_min_disc2(xk,fcp,gcp,X,Yk,rho,J):
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    p=0
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        p=cp.maximum(p,gi(x,param_gi))
    q=cp.hstack([gcp(x,Yk[j]) for j in J])
    prob = cp.Problem(cp.Minimize(fcp(x)+rho*cp.pos(cp.maximum(cp.max(q),p))),cons_x)
    prob.solve(solver)
    return prob.value,x.value

def compute_bf(xk,fcp,gcp,X,Yk,J):
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    for j in J:
        cons_x=cons_x+[0>=gcp(x,Yk[j])]
    prob = cp.Problem(cp.Minimize(fcp(x)),cons_x)
    prob.solve(solver)
    return prob.value,x.value

def compute_min_cutsurf(xk,fcp,gcp,X,Yk,U,s,J):
    n=np.size(xk)
    x = cp.Variable(n)
    sigma=cp.Variable(1)
    x0=cp.Variable(1)
    cons_sigma=[sigma>=0]
    cons_x0=[x0+sigma<=U[0]]+[fcp(x)<=x0]
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    for j in J:
        cons_sigma=cons_sigma+[gcp(x,Yk[j])+sigma*s[j]<=0]    
    prob = cp.Problem(cp.Maximize(sigma),cons_x+cons_x0+cons_sigma)
    prob.solve(solver)
    return sigma.value,x.value,x0.value

def compute_min_remez(xk,fcp,gcp,thetacp,regphicp,X,Yk,epsk,deltak,gammak):
    n=np.size(xk)
    x = cp.Variable(n)
    eta=cp.Variable(1)
    cons_eta=[eta>=0]
    cons_x=[]
    a=0
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        a+=thetacp(gi(x,param_gi)*deltak)/deltak
    cons_eta=[eta>=gammak/np.size(Yk)*(cp.sum([thetacp(gcp(x,yk)*deltak)/deltak for yk in Yk])+a)]
    prob = cp.Problem(cp.Minimize(fcp(x)+eta+epsk*regphicp(x)),cons_x+cons_eta)
    prob.solve(solver)
    return prob.value,x.value,eta.value

def prox(xk,hk,gammak,X):
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    for i in range(len(X)):
        u=X[i]
        cons_x=cons_x+[x[i]>=u[0]]+[x[i]<=u[1]]
    prob = cp.Problem(cp.Minimize(gammak*x.T@hk+1/2*cp.sum_squares(x-xk)),cons_x)
    prob.solve(solver)
    return x.value

def proximal_step(xk,h,gamma,X,hc=False):
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    if hc:
        cons_x+=[hc(x)<=0]
    prob = cp.Problem(cp.Minimize(h(x)+1/(2*gamma)*cp.sum_squares(x-xk)),cons_x)
    prob.solve(solver)
    return prob.value,x.value

def proximal_solver(xi,h,gamma,X,hc=False,tol=10**-3):
    start=time.time()
    for i in range(1000):
        xold=xi
        fi,xi=proximal_step(xi,h,gamma,X,hc)
        #print(np.linalg.norm(xold-xi,2))
        if np.linalg.norm(xold-xi,2)<=tol:
            end=time.time()
            print("Number of iter",i+1)
            print("Total time",end-start)
            print("Time per iteration",(end-start)/(i+1))
            break
    return fi,xi

def projection_step(xk,fk,v,gamma,X,hc=False): 
    n=np.size(xk)
    x = cp.Variable(n)
    cons_x=[]
    for i in range(len(X)):
        gi=X[i][0]
        param_gi=X[i][1]
        cons_x=cons_x+[gi(x,param_gi)<=0]
    if hc:
        cons_x+=[hc(x)<=0]
    prob = cp.Problem(cp.Minimize(fk+v.T@(x-xk)+1/(2*gamma)*cp.sum_squares(x-xk)),cons_x)
    prob.solve(solver)
    return prob.value,x.value

def projection_solver(xi,gamma,X,Yk,g,dg,f,df,rho=1,hc=False,tol=10**-3):
    start=time.time()
    for i in range(10000):
        ymax=Yk[np.argmax([g(xi,yk) for yk in Yk])]
        if g(xi,ymax)>0:
            v=df(xi)+rho*dg(xi,ymax)
            fi=f(xi)+rho*g(xi,ymax)
        else:
            v=df(xi)
            fi=f(xi)
        #print(fi)
        #print(g(xi,ymax))
        xold=xi
        xi=xold-gamma/(i+1)*v
        
        #fi,xi=projection_step(xi,fi,v,gamma/(i+1),X,hc)
        #print(fi,f(xi))
        #print(np.linalg.norm(xold-xi,2))
        #print(np.linalg.norm(xold-xi,2))
        print(np.linalg.norm(v,2))
        if np.linalg.norm(xold-xi,2)<=tol:
            end=time.time()
            #print(np.linalg.norm(xold-xi,2))
            print("Number of iter",i+1)
            print("Total time",end-start)
            print("Time per iteration",(end-start)/(i+1))
            break
        
    return f(xi),xi
    
def AlgPenDisc(x0,Y0,fcp,gcp,g,X,Y,rho=1,concave=False,gcps=[],drop_cuts=False,f=False,df=False,dg=False):
    proximal=False
    projection=True
    tol=10**-3
    n=np.size(x0)
    xk=x0
    Yk=Y0
    fk=f(xk)+rho*g(xk,Y0[0])
    print(fk)
    J=[i for i in range(len(Y0))]
    Jk=J
    xold=x0
    sigma=2
    gamma=1
    while_loops=100
    rho_inc=0
    inizio=time.time()
    chiamate_vf=0
    Upper_prob_time=0
    Lower_prob_time=0
    cuts=len(J)-1
    initialcuts=cuts
    for j in range(while_loops):
        timeUP=time.time()
        #print(Yk)
        #fk,xk,vfk=compute_min_disc(xk,fcp,gcp,X,Yk,rho,J)
        if proximal:
            def h(x):
                return fcp(x)+rho*cp.pos(cp.max(cp.hstack([gcp(x,yi) for yi in Yk ])))
            fk,xk=proximal_solver(xk,h,gamma,X)
        if projection:
            fk,xk=projection_solver(xk,gamma,X,Yk,g,dg,f,df,rho=rho)
        vfk=np.max([g(xk,yk) for yk in Yk])
        Upper_prob_time+=time.time()-timeUP
        if vfk>tol:
            print(rho)
            rho_inc=rho_inc+1
            rho=sigma*rho
        else:
            chiamate_vf+=1
            TimeLP=time.time()
            if concave==False:
                phi,yphi=vf(xk,g,Y)
            else:
                phi,yphi=vf_maximum_concave(xk,Yk[-1],gcps,Y)
            Lower_prob_time+=time.time()-TimeLP
            if phi<=tol:
                break
            else:
                    cuts+=1
                    Yk=np.append(Yk,[yphi],axis=0)
                    J=np.append(J,cuts)
                    if drop_cuts==True:
                        mask=np.array([g(xk,Yk[i])>-10**-2  for i in J])
                        J=np.array(J)[mask.flat]
                    else:
                        pass
    fine=time.time()
    tempo_esecuzione=fine-inizio
    print("Time", tempo_esecuzione)
    print("Time upper problem",Upper_prob_time)
    print("Time lower problem",Lower_prob_time)
    print("Feasibility", phi)
    print("Final rho", rho)
    print("Rho increases",rho_inc)
    print("Optimal value", fk)
    print("Value function call", chiamate_vf)
    
    return xk,fk,tempo_esecuzione,phi,j,rho_inc

def AlgBF(x0,Y0,fcp,gcp,g,X,Y,concave=False,gcps=[],drop_cuts=False):
    tol=10**-3
    gamma=1
    n=np.size(x0)
    xk=x0
    Yk=Y0
    J=[i for i in range(len(Y0))]
    Jk=J
    xold=x0
    while_loops=100
    inizio=time.time()
    chiamate_vf=0
    Upper_prob_time=0
    Lower_prob_time=0
    cuts=len(J)-1
    for j in range(while_loops):
        timeUP=time.time()
        def h(x):
            return fcp(x)
        def hc(x):
            return cp.max(cp.hstack([gcp(x,yi) for yi in Yk ]))
        fk,xk=proximal_solver(xk,h,gamma,X,hc)
        Upper_prob_time+=time.time()-timeUP
        TimeLP=time.time()
        if concave==False:
            phi,yphi=vf(xk,g,Y)
        else:
            phi,yphi=vf_maximum_concave(xk,Yk[-1],gcps,Y)
        Lower_prob_time+=time.time()-TimeLP
        if phi<=tol:
                break
        else:
            cuts+=1
            Yk=np.append(Yk,[yphi],axis=0)
            J=np.append(J,cuts)
            if drop_cuts==True:
                mask=np.array([g(xk,Yk[i])>-10**-2 for i in J])
                J=np.array(J)[mask.flat]
            else:
                pass
    fine=time.time()
    tempo_esecuzione=fine-inizio
    print("Time", tempo_esecuzione)
    print("Time upper problem",Upper_prob_time)
    print("Time lower problem",Lower_prob_time)
    print("Feasibility", phi)
    print("Optimal value", fk)
    print("Value function call", j+1)
    
    return xk,fk,tempo_esecuzione,phi,j

def AlgMePa(x0,Y0,fcp,gcp,g,X,Y,Ub=10000,concave=False,gcps=[],drop_cuts=False):
    tol=10**-3
    n=np.size(x0)
    xk=x0
    Yk=Y0
    J=[i for i in range(len(Y0))]
    xold=x0
    while_loops=100
    rho_inc=0
    inizio=time.time()
    chiamate_vf=0
    optimal_cuts=0
    U=np.zeros(n+1)
    U[0]=Ub
    sj=10**-3
    s=[sj for i in range(len(Y0))]
    sigmas=[Ub]
    beta=1.01
    Upper_prob_time=0
    Lower_prob_time=0
    for i in range(while_loops):
        timeUP=time.time()
        sigmak,xk,fk=compute_min_cutsurf(xk,fcp,gcp,X,Yk,U,s,J)
        Upper_prob_time+=time.time()-timeUP
        sigmas=np.append(sigmas,sigmak)
        TimeLP=time.time()
        if sigmak<=tol:
            break
        if concave==False:
            phi,yphi=vf(xk,g,Y)
        else:
            phi,yphi=vf_maximum_concave(xk,Yk[-1],gcps,Y)
        Lower_prob_time+=time.time()-TimeLP
        if phi>tol:
            Yk=np.append(Yk,[yphi],axis=0)
            s=np.append(s,[sj],axis=0)
            J=np.append(J,J[-1]+1)
        else:
            optimal_cuts=optimal_cuts+1
            U[0]=fk[0]
            U[1:]=xk
        if drop_cuts==True:
            D=[sigma>=sigmak*beta and sigmak*sj+g(xk,yphi)<0 for sigma in sigmas]
            print(sigmas>=sigmak*beta)
            print(sigmak*s+g(xk,yphi)<0)
            print(D)
    fine=time.time()
    tempo_esecuzione=fine-inizio
    print("Time", tempo_esecuzione)
    print("Time upper problem",Upper_prob_time)
    print("Time lower problem",Lower_prob_time)
    print("Feasibility", phi)
    print("Optimal value", fk[0])
    print("Optimality cuts", optimal_cuts)
    print("Feasibility cuts", i+1-optimal_cuts)
    print("Value function calls", i+1)
    
    return  xk,fk[0],tempo_esecuzione,phi,i,optimal_cuts,i-optimal_cuts

def AlgRemez(x0,Y0,fcp,gcp,g,X,Y,concave=False):
    def thetacp(x):
        return 1/2*cp.maximum(x,0)**2
   
    n=np.size(x0)
    xk=x0
    Yk=Y0
    xold=x0
    epsk=1
    deltak=1
    gammak=1
    
    while_loops=100
    inizio=time.time()
    chiamate_vf=0
    Upper_prob_time=0
    Lower_prob_time=0
    for j in range(while_loops):
        deltak=(j+1)**(5/2)
        gammak=(j+1)**(3/2)
        epsk=1
        timeUP=time.time()
        def regphicp(x):
            return cp.norm(x-xk,2)**2
        xold=xk
        fk,xk,vfk=compute_min_remez(xk,fcp,gcp,thetacp,regphicp,X,Yk,epsk,deltak,gammak)
        
        Upper_prob_time+=time.time()-timeUP
        TimeLP=time.time()
        if concave==False:
            phi,yphi=vf(xk,g,Y)
        else:
            phi,yphi=vf_concave(xk,yk,gcp,Y)
        Lower_prob_time+=time.time()-TimeLP
        if np.linalg.norm(xold-xk)<=10**-6:
            break
        else:
            Yk=np.append(Yk,[yphi],axis=0)
    fine=time.time()
    tempo_esecuzione=fine-inizio
    print("Time", tempo_esecuzione)
    print("Time upper problem",Upper_prob_time)
    print("Time lower problem",Lower_prob_time)
    print("Feasibility", phi)
    print("Optimal value", fk)
    print("Value function call", j)
    
    return xk,fk,tempo_esecuzione,phi,j

def AlgCoM(x0,y0,fcp,gcp,g,df,dg,X,Y,concave=False):
    n=np.size(x0)
    xk=x0
    Yk=[y0]
    J=[0]
    Jk=J
    eta=0
    xold=x0
    xbar=x0
    while_loops=100
    Xk=np.zeros((while_loops,n))
    Gammak=[0.05/(i+1)**(0.5) for i in range(while_loops)]
    Etak=[10/(i+1)**(0.5) for i in range(while_loops)]
    inizio=time.time()
    chiamate_vf=0
    Upper_prob_time=0
    Lower_prob_time=0
    for j in range(while_loops):
        TimeLP=time.time()
        if concave==False:
            phi,yphi=vf(xk,g,Y)
        else:
            phi,yphi=vf_concave(xk,Yk[j],gcp,Y)
        Lower_prob_time+=time.time()-TimeLP
        if phi<=Etak[j]:
            hk=df(xk)
            J=np.append(J,j)
            
        else:
            hk=dg(xk,yphi)
        timeUP=time.time()
        xk=prox(xk,hk,Gammak[j],X)
        Upper_prob_time+=time.time()-timeUP
        Xk[j]=xk
        Yk=np.append(Yk,[yphi],axis=0)
        xbar=np.sum([Xk[i]*Gammak[i] for i in J],0)/np.sum([Gammak[i] for i in J]) 
        fk=fcp(xbar)
       
    fine=time.time()
    tempo_esecuzione=fine-inizio
    print("Time", tempo_esecuzione)
    print("Time upper problem",Upper_prob_time)
    print("Time lower problem",Lower_prob_time)
    print("Feasibility", phi)
    print("Optimal value", fk)
    print("Value function call", j)
    
    return xk,fk,tempo_esecuzione,phi,j
