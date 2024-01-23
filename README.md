# LOD-net

###The diagnosis of colon polyps is important to the
prevention of colorectal cancer. Polyp segmentation, however, is
still a challenging problem given that recent medical computer-
aided equipment suffers from situations of polyp variations in
terms of size, color, texture, and poor illuminations brought by
collected endoscopy videos. These obstacles hinder the prediction
of the boundary of a polyp. Inspired by the observation that the
values of pixels on the border region change more sharply than
others, we propose our Oriented-Derivative (OD) representation
to capture the relationship between pixels and the boundary
region given distance and orientation. To adaptively use the pro-
posed representation in arbitrary frameworks, we design plug-in
modules to learn the representation and aggregate features for
improving the accuracy of boundary predictions in the polyp
segmentation task, which could be implemented in arbitrary
frameworks including the encoder-decoder and top-down archi-
tecture. Extensive experiment results show the improvement from
the proposed oriented-derivative representation for the polyp
segmentation task and the extendibility of our proposed modules
in different architectures. Our methods achieved an improvement
ranging from 0.3% to 2.5% (mDice) compared to the baseline on
five publicly available datasets including Kvasir, CVC-ClinicDB,
EndoScene, CVC-ColonDB, and ETIS.
