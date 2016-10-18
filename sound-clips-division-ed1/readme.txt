Using amplitude-divide method:
1. check the amplitude of the sould-clips, make samplified point with amplitude lower than a threhold value to be zero.
2. use "siginificant_piece" function, to check whether in an interval, there is enough non-trivial point (I hope to only preserve meaningful sound period).
3. have some overlap between adjacent intervals: make sure we get the full syllable, but become redundant.

Problem:
1. can not handle syllables which duration longer than expected.
2. hard to tell from liaison.

Log:
1. Spend much of time to test the reliability of this method :  barely satisfactory.
2. The duration of this method is fixed. Have tried to cut it into different-length period only according to amplitude, but failed: hard to determine the end condition.