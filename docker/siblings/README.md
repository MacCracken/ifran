# docker/siblings/

Drop the sibling binaries your jobs run here (attn11, tarka, anukūlana, …)
before building `docker/Dockerfile.trainer`. Everything in this directory
except this README is copied to `/opt/ifran/siblings/` in the image, which is
on `PATH`. A job spec names a binary by its full path, e.g.
`bin = "/opt/ifran/siblings/attn11"`.

The binaries themselves are not tracked: `.gitignore` ignores everything here
but this file.
