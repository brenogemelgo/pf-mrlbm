#pragma once

#if (defined(CASE_JET) + defined(CASE_STATIC_DROPLET) + defined(CASE_RTI)) != 1
#error "Select exactly one case: -DCASE_JET, -DCASE_STATIC_DROPLET, or -DCASE_RTI"
#endif

#if defined(CASE_JET)
#include "jet.cuh"
using SelectedCase = JetCase;
#elif defined(CASE_STATIC_DROPLET)
#include "staticDroplet.cuh"
using SelectedCase = StaticDropletCase;
#elif defined(CASE_RTI)
#include "rti.cuh"
using SelectedCase = RTICase;
#endif
