%% create swiss roll indexed by position along the roll
dat=[2+4*rand(1000,1), zeros(1000,1), 8*rand(1000,1)];
dat=sortrows(dat, 1);
[TH,R,Z] = cart2pol(dat(:, 1), dat(:, 2), dat(:, 3));
TH=TH+2*R;
[x y z]=pol2cart(TH,R,Z);
dat=[x y z];
QP(dat)
