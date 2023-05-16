# physics-project
Finding desired launch angle to hit target...

This probably isn't going to be useful for anything.
A simple, small project.

# GOAL
Given a starting velocity and a target distance, find the required launch angle.
This would be very straight forward if there was no air resistance as you could simply use a parabola.
However, with the addition of air resistance, simulating the "launch" with some sort of guesstimate then adjusting the angle and retrying is probably the best approach.

We ended up first simulating a bunch of angles until we find the furthest we can launch the projectile. From here we can check if the target is within reach. If it is, we can start from the closest we got to the target so far, that will be the guesstimate though I guess it's not really a guesstimate. ¯\\\_(ツ)\_/¯ Oh well. From here we can approach the required angle taking smaller and smaller steps until we fall within some tolerance distance away from the target.
Thus we have found our angle.
