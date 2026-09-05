# The hardest bug

A report that was fast for everyone except one customer, and only sometimes.

Every obvious explanation was wrong. The query plan looked fine in isolation.
The data volume was unremarkable. It reproduced perhaps one time in ten, which
is the worst possible frequency — often enough to matter, rarely enough that any
change I made looked like it had worked.

What eventually cracked it was giving up on reasoning and going to look. The
distribution of rows across the partitioning key was wildly uneven for that one
tenant, so a plan that was correct on average was catastrophic for them
specifically. The averages had been hiding it the whole time, and I had been
staring at averages because they were what the dashboard showed.

Two things stuck with me. First, that "sometimes" almost always means a
dimension I have not looked at yet — the bug is not random, my model is just
missing an axis. Second, that the fix was smaller than the investigation by an
order of magnitude, which is normal and which I now expect rather than resent.
