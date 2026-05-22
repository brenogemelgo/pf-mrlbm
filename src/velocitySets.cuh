#pragma once

#include "D3Q27.cuh"
#include "D3Q7.cuh"

// Hydrodynamic MR-LBM stencil.
using VelocitySet = D3Q27VelocitySet;

// Phase-field streaming stencil. Switch this line to D3Q27VelocitySet if desired.
using PhaseVelocitySet = D3Q27VelocitySet;

// Isotropic gradients/laplacians use the largest available stencil, independent of phase streaming.
using GradientVelocitySet = D3Q27VelocitySet;

static_assert(GradientVelocitySet::Q() >= VelocitySet::Q(), "GradientVelocitySet must be at least as large as the hydrodynamic stencil.");
static_assert(GradientVelocitySet::Q() >= PhaseVelocitySet::Q(), "GradientVelocitySet must be at least as large as the phase stencil.");
static_assert(GradientVelocitySet::max_abs_c() == VelocitySet::max_abs_c(), "Gradient and hydrodynamic stencils must use the same neighbor radius.");
static_assert(GradientVelocitySet::max_abs_c() == PhaseVelocitySet::max_abs_c(), "Gradient and phase stencils must use the same neighbor radius.");
