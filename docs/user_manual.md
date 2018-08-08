# ORSO User Manual

## Table of contents

* [Creating a user account](#creating-a-user-account)
* [Adding a dataset](#adding-a-dataset)
* [Dataset processing and network integration](#dataset-processing-and-network-integration)
* [Application views](#application-views)
  * [Home page](#home-page)
  * [Experiments tab](#experiments-tab)
    * [Experiment page](#experiment-page)
    * [All experiments page](#all-experiments-page)
    * [Recommended page](#recommended-page)
    * [Favorites page](#favorites-page)
    * [Your experiments page](#your-experiments-page)
  * [Users tab](#users-tab)
    * [User page](#user-page)
    * [All users page](#all-users-page)
    * [Following page](#following-page)
    * [Followers page](#followers-page)
  * [Explore tab](#explore-tab)
    * [Overview page](#overview-page)
    * [Dendrogram page](#dendrogram-page)
    * [Network page](#network-page)
    * [PCA page](#pca-page)

## Creating a user account

Many of ORSO's features are accessible without a user account. However, if a user account is created, ORSO will make personalized dataset recommendations based upon your activity. A user account is also required to add datasets to ORSO.

To create an account, simply click on the 'Sign Up' button on the top navigation bar. After providing your email address, username, and password, a user account will be created for you. You can also select whether or not your account will be made public to other users. You can change whether or not your profile is public at any time.

## Adding a dataset

After making a user account, datasets may be added by clicking on the 'Add new experiment' button under the 'Experiments' tab. To add a new dataset to the  network, ORSO requires annotated metadata and genome read coverage information.

After clicking the 'Add new experiment' button, you will be taken to a web form with fields for your experiment's metadata. The 'Cell type' and 'Target' fields refer to the cell line or tissue of the sample and the protein target of the experiment. The protein target varies from experiment to experiment. For instance, the protein could be targeted by an antibody in a ChIP-seq experiment or by an shRNA in a RNA-seq knockdown experiment. For target-less experiments, like standard RNA-seq or DNase-seq experiments, the 'Target' field should be left blank.

ORSO accepts read coverage information in bigWig format. For preparation of read coverage files, we recommend using [the standard protocols designed by the ENCODE consortium](#https://www.encodeproject.org/pipelines/). To provide read coverage information for your experiment, ORSO requires the bigWig file to be hosted on a publicly accessible web server. This requirement is similar to display requirements for the UCSC Genome Browser. If your institute does not offer web hosting services, we recommend following [UCSC's guidelines for hosting bigWig files](https://genome.ucsc.edu/goldenpath/help/hgTrackHubHelp.html#Hosting). After setting up your bigWig file for access by HTTP, simply provide the URL for the file and the aligned genome in fields under the Datasets heading. ORSO also supports stranded coverage information. To do this, you must provide two bigWig files, one for each strand.

To accommodate replicates, ORSO allows multiple Datasets to be associated with the same experiment. To add additional datasets, click on the 'Add additional dataset' button at the bottom of the page. After doing this, an additional set of fields will be added under the Datasets heading.

## Dataset processing and network integration

After a new experiment is submitted, ORSO will process that experiment's read coverage values and perform comparisons against its network considering annotated metadata and processed read coverage values.

ORSO maintains internal lists of genes and enhancers for each supported genome. Genes are taken from the [RefSeq annotation](https://www.ncbi.nlm.nih.gov/refseq/). The source of enhancer loci is different for each organism.  For human and mouse, enhancers are taken from the [VISTA database](https://enhancer.lbl.gov/). For *D. melanogaster*, enhancer loci are taken from [Kvon *et al*](http://enhancers.starklab.org/). For *C. elegans*, enhancer loci are taken from [Chen *et al*](https://www.ncbi.nlm.nih.gov/pubmed/23550086).

When an experiment is submitted, ORSO first finds the coverage at each gene and enhancer loci. For each gene, coverage is found at the promoter, across the entire gene model, and across only exons, giving three distinct values. These values are stored for comparisons with other experiments. ORSO also computes metaplots showing the average coverage across all genes and enhancers. These plots can be found in experiment page views.

The submitted experiment is then compared against others for integration into the ORSO network. Comparisons are made against each other hosted experiment with the same experiment type (RNA-seq, ChIP-seq, etc.). Metadata comparisons consider the cell type and protein target for each experiment. For cell type comparisons, the [BRENDA Tissue and Enzyme Source Ontology](https://www.brenda-enzymes.org/ontology.php?ontology_id=3) is used to group similar cell types. If two cell types have shared parents in the BRENDA ontology, they are considered similar. For protein target comparisons, the [STRING Database](https://string-db.org/) is used to find interacting proteins. If two proteins are found to interact, then the associated protein targets are considered similar.

Some experiments will only consider the cell type or protein target when accessing similarity while others will consider both. A list of relevant metadata fields for each experiment type are given below. If the contents of each relevant field are similar, then the two experiments will be connected in the ORSO network.

In read coverage comparisons, only loci relevant to the experiment type considered are used. For instance, only exon coverage across a gene model is used when comparing RNA-seq experiments. A complete list of relevant loci groups is given below.

To evaluate similarity, coverage values are applied to a model trained through three steps: (1) loci selection, (2) dimensionality reduction, and (3) a neural network classifier. Each step in the model was trained using the consortial ENCODE dataset.

Not all loci will be relevant to similarity comparisons. For example, the gene expression values for housekeeping genes are not generally informative for comparisons in RNA-seq data. Selection aims to limit analysis to informative loci. To do this, we train a Random Forest Classifier with loci coverage information to classify experiments by cell type and protein target. After fitting, we select the top 1000 loci by importance to the Random Forest model. Importance was evaluated using mean decrease impurity.

Selected loci are then applied to a PCA model for dimensionality reduction. We use this to reduce the dimensionality of loci coverage values from 1000 to 3. Three dimensional values were used because of their ease of display.

Lastly, the reduced values for each experiment are concatenated together and applied to a neural network classifier. The neural network classifier was trained with ENCODE data. For the training set, two experiments were considered similar if they shared attributes in the cell type or target fields. If the neural network classifies the two experiments as similar, then those experiments are connected in the network.

## Application views

The application views provide a web interface for exploring datasets hosted in the ORSO data network.

### Home page

On the Home page, there is a feed that reports the activity associated with followed users and favorited datasets. Here, you will find notifications about when your datasets are favorited or when followed users add new data.

### Experiments tab

Under the Experiments tab are views displaying experiment lists. Each list can be filtered by metadata and description fields.

#### Experiment page

Each hosted experiment has an individual page on ORSO. The Network tab gives a graph-based view of the connections between that experiment and others. The Metadata tab gives all the annotated metadata associated with an experiment. Under the Dataset Coverage tab, coverage for a selected dataset and a genomic loci group may be displayed.

Additional navigation options are also provided for the experiment. Using the 'Navigate To' dropdown, a list of similar experiments may be accessed. Links are also provided to pages for each dataset associated with the experiment.

#### All experiments page

This is a list of all public experiments. Each experiment is displayed using a panel. The star in the upper-right corner of each panel may be clicked to favorite the experiment.

#### Recommended page

This lists all experiments recommended to a user. Recommendations are based on ORSO network comparisons. The recommended list contains all experiments connected to experiments that you own or have favorited.

#### Favorites page

This lists all experiments that you have favorited. By clicking the star in the upper-right corner of a panel, you can un-favorite an experiment.

#### Your experiments page

This lists all your personal experiments. The 'pen' and 'trashcan' buttons can be used to update and delete an experiment, respectively.

### Users tab

Under the Users tab are views displaying lists of users with public accounts.

#### User page

Each user has a page summarizing their interactions with datasets and other users. The composition of a user's experiments is described in terms of the proportion of different cell types and protein targets. Navigation is also provided to a lists of personal experiments, followed users, and followers.

#### All users page

This lists all users with public accounts. This list can be searched by username. A user may be followed by clicking the star in the upper-right corner of each panel.

#### Following page

This lists all followed users. A user may be un-followed by clicking the star in the upper-right corner of a panel.

#### Followers page

This lists all of your followers. You can reciprocally follow another user by clicking the star in the upper-right corner of a panel.

### Explore tab

The Explore tab includes pages that give top-down views of the ORSO data network. With these views, dataset groups can be discovered and trends in data explored. Personal datasets can also be rapidly contextualized within the cohort of data hosted by ORSO.

#### Overview page

This view provides a simple overview of the experiments hosted by ORSO, including the proportion that come from ORSO users.

#### Dendrogram page

ORSO finds hierarchical groups of experiments by clustering datasets on the basis of their shared similarities. Clustering is performed by first generating a n-by-n similarity matrix, where n is the total number of experiments. With each row and each column corresponding to a dataset, intersections are set to 1 if the two datasets are similar, 0 if not. Hierarchical clustering is then applied to the similarity matrix, and the results are displayed using a dendrogram.

#### Network page

To show the connectivity between all datasets in the ORSO data network, ORSO displays the network in a graph view with datasets represented as nodes and similarities as edges. Again, these similarities are found based on comparisons of primary read coverage values and annotated metadata.

#### PCA page

Read coverage values are transformed by PCA to provide a quick means of comparing and contextualizing data. In this view, PCA-transformed values for each dataset are displayed in a three-dimensional view. To facilitate comparisons, user-directed commands allow selection of individual datasets and display colors. These settings are accessible in drop-down fields on the right side of the plot.
