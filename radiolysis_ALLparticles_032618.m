
clear
clc
w = warning ('off','all');


%1. Wavelength	nm
%2. photon energy  eV	
%3. solar flux	1/cm2/s/nm
%4. sig_c	cm2
%5. sig_d	cm2
%6. light penetration	A
%7. sig_c*SolarFlux	
%8. sig_d*SolarFlux	
%9. min(1,col(G)/10000)

SolarDat = load('CassidySolarF.txt');  % Saturn distance

%1. wavelength	nm
%2. n of ice	
%3. k of ice	
%4. absorbance	1/A
%5. absorption	cm2
%6. light penetration	A
%7. Photon Energy	eV
%8. Photon"stopping power" eV/A
%9. sigma_C	 cm2
%10. Gerakines source	eV/cm2/s/nm
%11. Gerakines source   photons/cm2/s/nm
%12. sigma_C	
%13. lampflux*sig_c	
%14. sig_d	
%15. sig_d	
%16. lampflux*sig_d	
%17. concentration

IceLampPrps = load('IceLampPrps.txt');
%set Gerakines Source
if 0
SolarDat(:,3) = interp1q(IceLampPrps(:,1),...
    IceLampPrps(:,11),...
    SolarDat(:,1));
end

semilogy(SolarDat(:,1),SolarDat(:,3),...
    SolarDat(:,1),SolarDat(:,4),...
    SolarDat(:,1),SolarDat(:,5))

photons = 0;
electrons = 1;
ions = 0;

qeff = 1;  %quantum efficiency
n = 3e22;  %1/cm3
sigD = 10^-14.1;  %cm2  general destruction CS
gma = 10^11.3;  %1/eV/cm2
T = 80;  %K  ice temperature
aa = 0.012;  %K-1
Gp0 = sigD*gma;  %eV-1  G-value at low T
Gp = Gp0*exp(-aa*T);  %eV-1  G-value
    a = (1e8*sigD/n)*gma*exp(-aa*T);%5e-18;%  %cm2A/eV   %times SP yields creation CS (not for photons)
xx = (0:100:3000000)*(1e-8);   % cm positions
dxx = (xx(3:end)-xx(1:end-2))/2;
dxx = [dxx(1), dxx, dxx(end)];

% sum over wavelength
ff = 0*xx;
create = ff;
destroy = ff;
dl = (SolarDat(3:end,1) - SolarDat(1:end-2,1))/2;
dl = [dl(1); dl; dl(end)];  %nm
for ii = 1:numel(SolarDat(:,1))
    if photons% && (SolarDat(ii,1) > 180) && (SolarDat(ii,1) < 200)
    A = SolarDat(ii,3);  %1/cm2/s/nm - input dstrb
    B = 1/(SolarDat(ii,6)*(1e-8));%n*SolarDat(ii,4)/qeff;  %1/cm
    ffnrm = B*exp(-B*xx);  %1/cm - normalized depth dstrb
    ff = A*ffnrm*dl(ii);  %1/cm3/s  photon dep rate density vs depth
    create = create + SolarDat(ii,4)*ff;  %1/cm/s
    destroy = destroy + SolarDat(ii,5)*ff;  %1/cm/s
    end
end

%loglog(xx,create./destroy)


%% electrons
if electrons
me = 9.11e-31;  %kg
c = 2.998e8;  %m/s

% e stopping power
load e_ranges  %A vs eV  in water ice
dE = e_ranges(:,1);
dE = (dE(3:end) - dE(1:end-2))/2;
dE = [dE(1); dE; dE(end)];
Drng = e_ranges(:,2);
Drng = (Drng(3:end) - Drng(1:end-2))/2;
Drng = [Drng(1); Drng; Drng(end)];
SP = dE./Drng;  %eV/A
%sigCe = a*SP;  %cm2  electron H2O2 creation CS
loglog(e_ranges(:,1),SP)


%%% electron distribution
%%% flux from all directions per m2 per s per eV, scale by 1/2 for projected cosine
% load Rhea_electrons_NoCosine
% E = Rhea_electrons(:,1);  %eV
% F = Rhea_electrons(:,2);  %1/m2/s/eV
load Enceladus_e_distributions
E = dstrbs.E_vals_eV';  %eV
F = permute(dstrbs.Edist_eV_matrix(1,1,:),[3 1 2]);  %1/m2/s/eV
rng = interp1q([0;e_ranges(:,1)],[e_ranges(1,2);e_ranges(:,2)],E);
SP = interp1q([0;e_ranges(:,1)],[SP(1);SP],E);
dE2 = (E(3:end) - E(1:end-2))/2;
dE2 = [dE2(1); dE2; dE2(end)];
% SP vs depth vs starting energy
% depth distribution of energy
% if 1
Evl = E;
SPvl = SP;
SPvls = SP*(0*xx);
for jj = 1:numel(xx)
    SPvls(:,jj) = SPvl;
    Evl = Evl - SPvl*(1e8*dxx(jj));
    %Evl(Evl<0) = 0;
%     [~,bb] = min(Evl(Evl > 0));
%     bb2 = bb + sum(Evl<=0);
%     Evl(bb2) = 0;
    %cnd = (Evl >= 0);
        %SPvl = interp1q([0;E],[SP(1);SP],Evl);
    SPvl = interp1q([E],[SP],Evl);
    SPvl(isnan(SPvl)) = 0;
    if ~mod(jj,1000)
        jj/numel(xx)
    end
end
% save SPvlsB SPvls
% else
%     load SPvlsB
% end
%image(3e2*SPvls)


% loop over distribution
% create = 0*create;
% destroy = 0*destroy;
%Anrm = sum(F.*dE2)/1e4;  %1/cm2/s  total particle Flux
for ii = 1:numel(F)   
    A = F(ii)/1e4;  %1/cm2/eV/s   particle flux per eV
    A = A.*(SPvls(ii,:) > 0);  % cuts part of distribution that does not reach depth
    Prd = Gp*A.*SPvls(ii,:)*(1e8);  %1/cm3/s/eV  peroxide production rate per unit volume PER eV (vs depth)
    create = create + Prd*dE2(ii);  %1/cm3/s  peroxide production rate per unit volume VS DEPTH
    destroy = destroy + sigD*A*dE2(ii);  %1/s  destruction rate per peroxide molecule
end

end
loglog(xx/100,create./destroy)
axis([1e-8 1e-3 1e-8 1e-1])


%% ions
if ions
%e = 1.60217646e-19;  % J
Angle = 0;
Eset = 300;  %eV

%  Projectile properties
species = 'O';
if strcmp(species,'H')
    load H_stopping
    load H_ranges
    ranges = H_ranges;
    m1 = 1;  %  amu
    Z1 = 1;
elseif strcmp(species,'O')
    load O_stopping
    load O_ranges
    ranges = O_ranges;
    m1 = 16;  %  amu
    Z1 = 8;
elseif strcmp(species,'S')
    load S_stopping
    load S_ranges
    ranges = S_ranges;
    m1 = 32;  %  amu
    Z1 = 16;
elseif strcmp(species,'Ar')
    load Ar_stopping
    load Ar_ranges
    ranges = Ar_ranges;
    m1 = 40;  %  amu
    Z1 = 18;
end
%filename = ['iso_yield_', species];
StoppingTable = StoppingTable_struct.IonizationTable' + StoppingTable_struct.RecoilTable';
% stopping_interp = interp1(StoppingTable_struct.Energies, StoppingTable, Eset);
% stopping_interp = [StoppingTable_struct.depth_values'*cos(Angle), stopping_interp'/cos(Angle)];



%%% load Ion distribution %%%
load Enceladus_O_distributions_gyro
E = dstrbs.E_vals_eV';  %eV
F = permute(dstrbs.Edist_eV_matrix(1,1,:),[3 1 2]);  %1/m2/s/eV
rng = interp1q([0;ranges(:,1)],[ranges(1,2);ranges(:,2)],E);
dE2 = (E(3:end) - E(1:end-2))/2;
dE2 = [dE2(1); dE2; dE2(end)];

SPvls = interp1q((1e-8)*StoppingTable_struct.depth_values',...
    StoppingTable', xx')';
SPvls = interp1q([0;StoppingTable_struct.Energies;E(end)],...
    [SPvls(1,:);SPvls;SPvls(end,:)], E);
SPvls(isnan(SPvls)) = 0;
% loop over distribution
% create = 0*create;
% destroy = 0*destroy;
%Anrm = sum(F.*dE2)/1e4;  %1/cm2/s  total particle Flux
for ii = 1:numel(F)   
    A = F(ii)/1e4;  %1/cm2/eV/s   particle flux per eV
    A = A.*(SPvls(ii,:) > 0);  % cuts part of distribution that does not reach depth
    Prd = Gp*A.*SPvls(ii,:)*(1e8);  %1/cm3/s/eV  peroxide production rate per unit volume PER eV (vs depth)
    create = create + Prd*dE2(ii);  %1/cm3/s  peroxide production rate per unit volume VS DEPTH
    destroy = destroy + sigD*A*dE2(ii);  %1/s  destruction rate per peroxide molecule
end

end

np = create./destroy;  %cm-3  equilibrium peroxide density
conc = np/n;
loglog(xx, conc)  %VS cm
loglog(xx, create/2e17, xx, destroy)  %VS cm
%axis([1e-8 1e-3 1e-8 1e-1])
sum(np.*dxx)  %per cm2


%%% Testing Burial rate time constant for Christine - 032618
% -burial*dCdx = destroy
burial = 100*1.6e-11;  %cm/s
    %%% SMOOTH %%%
    for ss=1:1
        conc = (conc(2:end) + conc(1:end-1))/2;
        conc = (conc(2:end) + conc(1:end-1))/2;
        conc = [conc(1), conc, conc(end)];
    end
    %%%%%
dC = (conc(3:end)-conc(1:end-2))/2;
dC = [dC(1), dC, dC(end)];
dCdx = dC./dxx;
y1 = -burial*dCdx;  %1/s
y2 = destroy;  %1/s
loglog(xx, y1, xx, y2)  %VS cm
% [~,bb] = min(abs(y2-y1));
% conc(bb)
% xx(bb)


%%% Modify conc with burial rate
dltaMin = 1;
dlta = 0;
cnt = 1;
while 0%(dlta > dltaMin) || (cnt<=1)
    conc0 = conc;
    
    dC = (conc(3:end)-conc(1:end-2))/2;
    dC = [dC(1), dC, dC(end)];
    dCdx = dC./dxx;
    
    destroyEff = destroy + burial*dCdx;  %1/s "effective" destruction rate
    np = create./destroyEff;
    concNEW = np/n;
    
    fct = 0.2;
    Log_conc = fct*log(concNEW) + (1-fct)*log(conc0);
    conc = exp(Log_conc);
    
    loglog(xx, conc)  %VS cm
    axis([1e-6 1e-0 1e-10 1e0])
    pause(0.01);
    

    cnt = cnt + 1;  
end


