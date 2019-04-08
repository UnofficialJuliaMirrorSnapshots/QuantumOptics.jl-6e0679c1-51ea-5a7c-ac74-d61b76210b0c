using Test
using QuantumOptics

@testset "schroedinger" begin

N = 3
Ncutoff = 2
T = [0.:0.1:1.;]

Ω = [0 2 3;
     2 0 1;
     3 1 0]

ω = [1., 1.2, 1.5]

basis_fock = FockBasis(Ncutoff)
basis = tensor([basis_fock for i=1:N]...)

a = destroy(basis_fock)
at = create(basis_fock)
I = identityoperator(basis_fock)


psi0 = tensor([coherentstate(basis_fock, i%Ncutoff) for i=1:N]...)


# Interaction picture
Hrot = SparseOperator[]
for i=1:N, j=1:N
    if i==j
        continue
    end
    h = embed(basis, [i,j], [a, Ω[i,j]*at])
    push!(Hrot, h)
end
Hrot = sum(Hrot)

# Schroedinger picture
function f(t, psi)
    H = SparseOperator[embed(basis, i, ω[i]*at*a) for i=1:N]
    for i=1:N, j=1:N
        if i==j
            continue
        end
        h = embed(basis, [i,j], [a, exp(1im*(ω[i]-ω[j])*t)*Ω[i,j]*at])
        push!(H, h)
    end
    sum(H)
end

tout, psi_rot_t = timeevolution.schroedinger(T, psi0, Hrot)
tout, psi_t = timeevolution.schroedinger_dynamic(T, psi0, f)

n_op = dense(at*a)
for (i, t) in enumerate(tout)
    R = prod([embed(basis, i, exp(1im*ω[i]*t*n_op)) for i=1:N])
    psi_rot = psi_rot_t[i]
    psi = psi_t[i]
    # @test abs(dagger(psi_rot)*R*psi) < 1e-5
    rho = dm(psi)
    rho_rot = dm(psi_rot)
    @test tracedistance(rho_rot, dense(R)*rho*dagger(dense(R))) < 1e-5
end

function fout(t, psi)
 deepcopy(psi)
end
t_fout, psi_fout = timeevolution.schroedinger(T, psi0, Hrot; fout=fout)
@test t_fout == tout && psi_fout == psi_rot_t

end # testset
