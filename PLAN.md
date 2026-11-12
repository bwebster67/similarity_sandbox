
# Is disagreement between subjects due to multiple equally valid groupings, or due to noise?

$H_0$: Variance is due to noise.
$H_1$: There are multiple valid groupings. (Multimodal)
$H_2$: There are no obvious groupings. (Uniform)

### Data:
SPoSE-49, SPoSE-66, SRF, CLIP
Human Consistency 
VLM responses


## Ben's To-do (By Monday!)

- [x] Write a function to calculate probability estimates from embeddings.
    -  For a triplet, compute cosine distance for each pair. Normalize distance. Sum up the total. Divide each cosine distance by the total.  
- [ ] Write a function to calculate consistency from human responses as described below.
- [ ] Estimate response likelihood using multiple estimates of image similarity. Save each to a csv. 
    - CLIP
    - SPoSE-49
    - SPoSE-66
    - SRF
- [ ] Measure correlation between each of the methods.



## Next:
- [ ] Calculate consistency as descrbied below. Get rid of the current method
- [ ] Rank triplets with 5+ responses by the highest percentage value of the three
- [ ] Plot the distribution!
- [ ] Obtain VLM responses for the 2000 triplets (5+ responses) 
- [ ] Compute correlation between VLM responses and humuan consistency values

## Then:
- [ ] Consider estimations of similaritiy between images. (cos dist. of CLIP, SPoSE-49, SPoSE-66, SRF)
- [ ] Analyze explanations from VLM
- [ ] Compute correlation between all the different percentage distributions.

*We believe this is the first time variance has been looked at per-triplet*

#### Getting Consistency from human responses
For each triplet (say 100 people responded)
add up how many people chose each as the odd one out
and divide each by the number of people who responded.

This way you get a response similar to the VLM!
save this as a CSV.


we have cos dist for 3 pairs (1 triplet)
sum the distances into one
take each distance and we divide it by the sum, we turn it into a ratio