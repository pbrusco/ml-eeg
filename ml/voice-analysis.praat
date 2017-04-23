# This script extracts jitter, shimmer and NHR from an interval of a given file.
# Agustin Gravano March 2008
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

#############################################################
# Extract voiced intervals from a selected Sound object
# (based on code by John Tondering, Niels Reinholt Petersen),
# and compute jitter and shimmer from the voiced intervals.

select Sound sound_all
To Pitch (ac)... 0.01 'min_pitch' 15 no 0.03 0.7 0.01 0.35 0.14 'max_pitch'

median_f0 = Get quantile... 0 0 0.5 Hertz
mean_period = 1/median_f0
number_of_voiced_frames = Count voiced frames

if number_of_voiced_frames > 0
	select Sound sound_all
	plus Pitch sound_all

	To PointProcess (cc)
	To TextGrid (vuv)... 0.02 mean_period
    Rename... sound_all

	select Sound sound_all
	plus TextGrid sound_all
	Extract intervals... 1 no V
	numberOfSelectedSounds = numberOfSelected ("Sound")
	Concatenate

	for i from 1 to 'numberOfSelectedSounds'
	slet_fil$ = "sound_all_V_'i'"
	select Sound 'slet_fil$'
	Remove
	endfor

	select Sound chain
	Rename... sound_voiced
	select PointProcess sound_all
	Remove

	call jitter_and_shimmer sound_voiced
else
	printline sound_voiced_local_jitter:--undefined--
	printline sound_voiced_local_shimmer:--undefined--
endif

############################################################
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
