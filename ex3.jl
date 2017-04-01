# This program simulates the motion of stars in a 1D universe.
 

# Information about a Star in a 1D universe
type Star
    mass # Mass
    pos  # Coordinate
    vel  # Velocity
end

# Global variable univ is a Vector{Star}, declared much later.

# Return array of forces acting on each star in univ
function compute_force()
    n = length(univ)
    force = zeros(n)
    for i=1:n, j=1:n
        force[i] += (univ[i].mass*univ[j].mass)*sign(univ[j].pos-univ[i].pos)
    end
    force
end

# Update state of each star in univ
function apply_force!(force, dt)
    for i=1:length(univ)
        acc = force[i]*dt/univ[i].mass
        univ[i].pos += (univ[i].vel + 0.5*acc*dt)*dt
        univ[i].vel += acc*dt
    end
end

# Simulate one time unit using m steps
function step(m)
    dt = 1/m
    for k=1:m
        force = compute_force()
        apply_force!(force, dt)
    end
end

# DO NOT CHANGE CODE BELOW THIS LINE ------------------

# Make runs repeatable
srand(42)

# A universe of stars with random initial conditions
univ = Star[Star(rand(),rand(),rand()) for i=1:100]

@time step(1)
@time step(1000)
println(univ[50])

