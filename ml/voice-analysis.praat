# This script extracts jitter, shimmer and NHR from an interval of a given file.
# Agustin Gravano March 2008
# Modified by Pablo Brusco (March 2017)
#

############################################################
#  Get parameters from command line

form Enter Info
  word file /proj/corpora/games/data/session_02/s02.objects.1.A.wav
  real start_point 22.9225
  real end_point 23.6525
  real min_pitch 100
  real max_pitch 500
endform

############################################################
#  Open sound file, extract portion specified,
#  and compute jitter and shimmer.

Open long sound file... 'file$'
Rename... long_sound
Extract part... 'start_point' 'end_point' no
Rename... sound_all

call jitter_and_shimmer sound_all

############################################################
#  Compute the Noise-to-Harmonics Ratio (NHR)

select Sound sound_all
plus Pitch sound_all
plus PointProcess sound_all

voiceReport$ = Voice report... 0 0 'min_pitch' 'max_pitch' 1.3 1.6 0.03 0.45
nhr = extractNumber (voiceReport$, "Mean noise-to-harmonics ratio: ")
printline noise_to_harmonics_ratio:'nhr:6'


############################################################

procedure jitter_and_shimmer sound$

select Sound 'sound$'
To Pitch (cc)...  0 'min_pitch' 15 no 0.03 0.45 0.01 0.35 0.14 'max_pitch'
To PointProcess

print 'sound$'_local_jitter:
j=Get jitter (local)... 0 0 0.0001 0.02 1.3
printline 'j:6'

select Sound 'sound$'
plus PointProcess 'sound$'

print 'sound$'_local_shimmer:
s=Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
printline 's:6'

endproc
