function [ReconImage, support] = Reconstruction_v3(Data,lambda,Dz)

delta2 = 2.2e-6;        % [m] pixel size of CCD
  
k = 2*pi/lambda;    % wavevector

% Force the upsampling factor to 2
UpsampleFactor = 2;

% When backround intensities are different between sample and ref images
%  bgData = mean2(Data);
%  bgRef = mean2(Ref);
%  NormFactor = bgRef/bgData;

% subNormAmp = sqrt(Data(miny:maxy,minx:maxx)./(Ref(miny:maxy,minx:maxx)));
subNormAmp = sqrt(Data);
        
delta_prev = delta2;

if UpsampleFactor > 0
    [subNormAmp,delta2] = upsampling(subNormAmp,UpsampleFactor,delta_prev);    
end

Nx = size(subNormAmp,1);
Ny = size(subNormAmp,2);
delta1 = delta2;

dfx = 1/(Nx*delta2);  % grid spacing in frequency domain
dfy = 1/(Ny*delta2);

[fx fy] = meshgrid((-Ny/2:Ny/2-1)*dfy,(-Nx/2:Nx/2-1)*dfx);


% Transformation function
Gbp = zeros(Nx,Ny);
Gfp = zeros(Nx,Ny);
for n = 1:size(fx,1)
    for m=1:size(fx,2)
        Gbp(n,m) = exp(1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
        Gfp(n,m) = exp(-1i*k*Dz*sqrt(1-lambda^2*fx(n,m)^2-lambda^2*fy(n,m)^2));
    end
end


% The number of iteration
NumIteration = 30;   

% The measured hologram amplitude is used for an initial estimate
Input = subNormAmp;


%% Reconstruction process

for k=1:NumIteration
    
    % Fourier transform of function at the detector plane. 
    F2 = ft2(Input,delta2);      

    % Reconstructed image at the object plane
    Recon1 = ift2(F2.*Gbp,dfx,dfy);

    
%% Object support
    if k==1

        
          Threshold_objsupp = 0.04; 
%               support = stdfilt(abs(Recon1),ones(9));
          support = stdfilt(abs(Recon1).*cos(angle(Recon1)),ones(9));

          support = im2bw(support,Threshold_objsupp);
          se = strel('disk',6,0);
          support = imdilate(support,se);
          support = imfill(support,'holes');
          support = bwareaopen(support,300);


%           support = stdfilt(abs(Recon1),ones(9));
%           support = im2bw(support,0.05);
%           se = strel('disk',6,0);
%           support = imdilate(support,se);
%           support = imfill(support,'holes');
%           support = bwareaopen(support,300);
            
    end
   
%% Constraint
    
    % Preserve images inside the object support and set "1" to the
    % pixels outside of the object support
            
    Constraint = ones(size(Recon1));
    
    for p=1:size(Recon1,1)
        for q=1:size(Recon1,2)
            
            if support(p,q) == 1
                Constraint(p,q) = abs(Recon1(p,q));
            end
            
    % Transmission constraint
    % if the transmission is greater than 1, the value is set to unity. 
    % basic assumption is the normalized transmission (absorption) value
    % cannot be greater (lower) than 1 (0). If it happens, it is due to
    % inteference with its twin image. 

            if abs(Recon1(p,q))>1
                Constraint(p,q) = 1;                
            end
        end
    end
    

% togglePhase flipping 
% Zhao et al., Opt. Eng. 50, 091310 (2011)
% flip the togglephase of object by changing the sign of the exponential term
%     if k==1
%         Recon1_update = Constraint.*exp(-1i*angle(Recon1));
%     else
%         Recon1_update = Constraint.*exp(1i*angle(Recon1));
%     end
    
    Recon1_update = Constraint.*exp(1i*angle(Recon1));
    

    % Fourier transform of function at the object plane for transformation
    F1 = ft2(Recon1_update,delta1);
    
    % Reconstructed image at the detector plane
    Output = ift2(F1.*Gfp,dfx,dfy);

    % New input for the next iteration
    Input = subNormAmp.*exp(1i*angle(Output));
    
    
    % For the last iteration
    if k==NumIteration
        Output_Final = Input;
    end    
end

F2 = ft2(Output_Final,delta2);
ReconImage = ift2(F2.*Gbp,dfx,dfy);





    
    
    
    
