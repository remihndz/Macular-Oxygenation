reset
set terminal png enhanced
set encoding utf8

datafile = ARG1.ARG2

# Profile in the parafovea
set output ARG1.'parafoveal_profile.png' 
set title "PO2 at x=50μm (parafoveal region)"
set xlabel "Distance from the choroid (μm)"
set ylabel "PO2 (mmHg)"
plot datafile u 1:2 w l notitle

# Profile in the transition region
set output ARG1.'transition_zone_profile.png' 
set title "PO2 at x=750μm (transition region)"
set xlabel "Distance from the choroid (μm)"
set ylabel "PO2 (mmHg)"
plot datafile u 1:3 w l notitle

# Profile at the fovea
set output ARG1.'foveal_profile.png' 
set title "PO2 at x=1100μm (fovea)"
set xlabel "Distance from the choroid (μm)"
set ylabel "PO2 (mmHg)"
plot datafile u 1:4 w l notitle

# Profile at the FAZ
set output ARG1.'faz_profile.png' 
set title "PO2 at x=1500μm (foveal avascular zone)"
set xlabel "Distance from the choroid (μm)"
set ylabel "PO2 (mmHg)"
plot datafile u 1:5 w l notitle

# Convergence rate
set output ARG1.'FixedPointConvergence.png'
set title "Convergence rate of the fixed point iteration"
set xlabel "Number of iterations"
set ylabel "Normalized error (log scale)"
set logscale y
plot ARG1."Convergence.dat" w l lt rgb "black" dt 4 t "L2 error of fixed point iteration", exp(-x/3) ls 0 lt rgb "black" t "exp(-x/3)"