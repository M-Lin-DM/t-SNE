function out=norm_array(A)%A is NxM  of Mdimensional row vectors
out=sqrt(sum(A.^2,2));
end
%finds the norm of each row in an array assuming rows are vectors