clear all;
addpath('mackeyglass_jst');             % to add test functions 

rng(12345);                              % rand seed

%% ==================================== PARAMETERS ====================================
SLM_Nx=2^5;                             % number of x pixels in SLM
SLM_Ny=2^5;                             % number of y pixels in SLM
M=SLM_Nx*SLM_Ny;                 % total number of pixels on SLM
RC_N=M;                          % number of nodes in the reservoir

T1=100;                                 % number of init timesteps
T2=1000;                                % number of training timesteps
T3=1000;                                % number of test timesteps
Ttot=T1+T2+T3;                          % total timesteps

alpha=1;
%% ==================================== VARIABLES ====================================
bias=1;
RC_TM=exp(1i*2*pi*rand(RC_N))/sqrt(RC_N);          % linear transfer matrix of Reservoir
CCD_max=0.5*max(abs(RC_TM*ones(RC_N,1)).^2);% normalization factor for the CCD camera (full field transmission)

fB=1/10; % bias part of the SLM
fRC=7/10;% network part of the SLM
fU=1-fB-fRC; % data part of the SLM

MB=round(fB*M);                    % (a) portion of the SLM set for bias
MRC=round(fRC*M);                  % (b) portion of the SLM set for network
MU=M-MB-MRC;                       % (c) portion of the SLM set for data

RC=rand(MRC,1);                % (b) portion of the SLM containing the network
idx1=1:MB;                             % indes to select (a)
idx2=(1+MB):MRC;                  % index to select (b)
idx3=(1+MB+MRC):M;                % index to select (b)

Win=rand(length(idx1),1);        % binary input weights for the data

%% ==================================== STORAGE VECTORS ====================================
Em=zeros(M,1);                   % init RC input
Ep=zeros(M,1);                   % init RC output
CCD=zeros(M,1);                  % init CCD output

M_train=zeros(M,T2);                   % init storage matrix for training stage
M_test=zeros(M,T3);               % init storage matrix for test stage

%% ==================================== INPUT DATA ====================================
fRC=10;                                 % interpolation factor (see lit.)
Nt=fRC*(Ttot);                          % number of timesteps (total);
x0=1.8;                                 % initial condition (lit.)
dt=0.1;                                 % time-step (lit.)
u=mackeyglass(Nt,x0,dt);                % create mackeyglass sequence
u=u(1:fRC:end-1);                       % interpolate

u=u-min(u)+1;                            % data normalization (for stability)
u=u/max(u);                             % final normalized data (reay for training)

Nfreq=2;
u=0;
for id=1:Nfreq
    u=u+1.2+0.25*sin(500.3*rand(1)*pi*(1:(Ttot))/Ttot);
end
u=u(:)/Nfreq;

%% =====================================================================================
%% =====================================================================================
%% ==================================== STARTUP STAGE ==================================
for t=1:T1
    Em(idx1)=u(t).*Win;
    Em(idx2)=RC;
    Em=exp(1i*pi*Em);
    Ep=prop(RC_TM,Em,alpha);                  % Ep=RC_TM*Em;   
    CCD=act(Ep)/CCD_max;
    RC=CCD(idx2);
end
%% ==================================== TRAINING STAGE =================================
for t=1:T2
    Em(idx1)=u(t+T1).*Win;
    Em(idx2)=RC;
    Em=exp(1i*pi*Em);
    Ep=prop(RC_TM,Em,alpha);                  % Ep=RC_TM*Em;   
    CCD=act(Ep)/CCD_max;
    RC=CCD(idx2);
    M_train(:,t)=CCD;
end
%% ==================================== RIDGE REGRESSION ===============================
X=transpose(M_train);                         % regression matrix [Nt,M]
Y_train=u((T1+2):(T1+T2+1));                  % target data (predict next time step)

% [Wout,beta0]=rr_builtin(X,Y_train);           % regression with MATLAB ridge

[Wout,beta0,lambda_opt]=rr_optimised(X,Y_train);         % Tikhonov regression with optimised ridge parameter

%[Wout,beta0,F]=rr_cv(X,Y_train,10,lambda_opt);         % regression with MATLAB functions (include CV)

Yhat_train=X*Wout+beta0;         % best predicted data
fom_train=nmse(Yhat_train,Y_train);                % FOM for prediction

%% ==================================== TEST STAGE =====================================
Y_test=[u(T1+T2+(2:T3));0];
Yhat_test=zeros(T3,1);

y0=u(1+T1+T2);
for t=1:T3
    %Em(idx1)=u(t+T1+T2).*Win;
    Em(idx1)=y0.*Win;
    Em(idx2)=RC;
    Em=exp(1i*pi*Em);
    Ep=prop(RC_TM,Em,alpha);                  % Ep=RC_TM*Em;   
    CCD=act(Ep)/CCD_max;
    RC=CCD(idx2);
    M_test(:,t)=CCD;
    y0=transpose(CCD)*Wout+beta0;
    Yhat_test(t)=y0;
end

fom_test=nmse(Yhat_test,Y_test);
%% ==================================== RMS ============================
function y=nmse(X1,X2)
    a=norm(X1-X2)^2;
    b=norm(X2)^2;
    y=a/b/length(X1);
end
%% ==================================== ACTIVATION FUNCTION ============================
function y=act(X)
    y=abs(X).^2;
    %y=abs(X);
    %y=tanh(abs(X).^2);
end

%% ==================================== FORWARD PROP STEP =============================
function xp=prop(W,xm,alpha)
    xp=W*xm;
    %xp=alpha*W*xm + (1-alpha)*xm;
end

%% ==================================== Ridge Regression builtin=======================
function [Wout,beta0]=rr_builtin(X,Y)
    k=0:1e-5:5e-3;                          %
    k = logspace(-6,-1,15);
    k=k(1);                                 % 

    B = ridge(Y,X,k,0);                     % scaled = 0 to 

    Wout=B(2:end,1);                   
    beta0=B(1,1);
end
%% ==================================== Ridge Regression builtin=======================
function [Wout,beta0,lambda_opt]=rr_optimised(X,Y)
    X=[X,ones(size(X,1),1)];
    lambda=1e-4;                              % ridge regularization term
    sX=size(X,2);
    Xt=transpose(X);
    
    myoptim=optimset('Display','iter','TolFun',1e-10,'TolX',1e-10);
    [lambda_opt,mymse]=fminsearch(@(x) myopt(x,X,Xt,sX,Y),lambda,myoptim);
    tmp1=inv(Xt*X+lambda_opt*eye(sX));    % inverse matrix
    Wout=tmp1*Xt*Y;          % one-step ridge regression
    beta0=Wout(end);
    Wout=Wout(1:(end-1));
end
function mse=myopt(beta,X,Xt,sX,Y)
    tmp1=inv(Xt*X+beta*eye(sX));        % inverse matrix
    Wout=tmp1*Xt*Y;          % one-step ridge regression 
    Yhat=X*Wout;             % output readout channels
    mse=norm(Yhat-Y).^2/norm(Y).^2/length(Y);
end

%% ==================================== Ridge Regression builtin=======================
function [Wout,beta0,F]=rr_cv(X,Y,mycv,lambda)

    [B,F]=lasso(X,Y,'CV',mycv,'Alpha',1e-13,'Lambda',lambda);
    [mm,im]=min(F.SE);
    Wout=B(:,im);                   
    beta0=F.Intercept(im);
end



