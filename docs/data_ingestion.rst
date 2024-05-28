Data Ingestion
==============

Getting data into LBANN requires using one of the predefined data
readers or writing a customized data reader tailored to the data in
question. Currently, data readers serve two roles:

1. Define how to ingest data from storage (at rest) and place it into
   an LBANN-compatible format

2. Understand the structure of a well-defined ("named") data set such
   as MNIST or ImageNet-1K (ILSVRC).

As LBANN is evolving, we are working to separate these two behaviors
into distinct objects, but it is still a work in progress.  As a
result there are some "legacy" data readers that represent both of
these features, and some new data readers that focus more on task 1
and incorporate the use of a sample list to help with task 2.

At this time, LBANN can only ingest static data sets; work on
streaming data is in progress.


Legacy Data Readers
-------------------

Some of the legacy data readers are the ``MNIST``, ``ImageNet``, and
``CIFAR10`` data readers.


"New" Data Readers
-------------------

Two of the new format data readers are the ``python``, ``SMILES``, and
:ref:`HDF5<sec:hdf5_data_reader>` readers.

Several of these readers (SMILES and
:ref:`HDF5<sec:hdf5_data_reader>`) support the use of :ref:`sample
lists<sec:sample-lists>`.

Iterative algorithms and data ingestion
---------------------------------------
One of the challenges of data ingestion is managing the interplay
between the size of the data set and the size of the mini-batch
consumed by each step of the execution algorithm.  With respect to the
model and execution algorithm there are two key fields that capture
this information:

1) the maximum mini-batch size (`max_mini_batch_size`) that a model is
   configured to support.  Nominally this is dictates how much memory
   is allocated in each tensor, and is established and then allocated
   during the model setup.  It is a property of the model.  Note that
   at the current moment, if the current mini-batch size exceeds the
   maximum mini-batch size, a warning is thrown and then the matrices
   are resized.

2) the current mini-batch size (`current_mini_batch_size`), which is
   dictated by how much data is available from the data ingestion
   pipeline.  This value can vary from step to step, but typically is
   equal to the maximum mini-batch size for all but the last step of
   an execution algorithm (when the data ingestion pipeline has run
   out of data).  The field for the current mini-batch size is
   governed by the data readers and then is cached in both the model
   as well as the current execution context.  Note that it is not
   clear if the execution contexts should hold this data anymore.


"Really New" Data Subsystem
---------------------------

During execution LBANN will ingest one or more streams of data.  There
will be unique streams of data for each execution mode:
 - training
 - validation
 - tournament
 - testing
 - inference

Note that execution modes should become more flexible and should be
able to be arbitrarily named.

The data stream object is responsible for keeping track of the "count"
/ state of that data stream for that execution context.  For bounded /
batched data streams, this would be the current position within the
stream and the total number of passes over the stream. (index and
epoch)

For infinite streams the object will just maintain the index /
position within the stream.

In both cases it is necessary for the object to track the "step" size
(i.e. mini-batch size).  Additionally, because the data stream will be
accessed in parallel, it is necessary to track the position of each
rank within the stream in terms of offset.

..
   Data source class file:  The data source class tracks the statefule
   aspects of one logical stream of data.
   Data sources are either bounded or infinite
   data sources.  The class is responsible for keeping track of state
   with respect to

Sample list:

Track how to retrive a data set from the outside world.  This
typically is a set of file locations for each sample as well as a
count of how many samples are in the set.

Data coordinator:

Responsible for managing one or more data streams for each execution
context.  It is


data reader / loader:

Function to ingest bits from outside and place them into an in-memory
object that is managed by the data coordinator.

Data store:
in-memory data repository for holding samples that have been read in

io_data_buffer:
Holds sample being fetched or the future of it.

data packer:
copies data fields from conduit nodes and maps them to Hydrogen
matrices.  Specific to a data set

Data Set:
The dataset class currently holds the number of samples processed, the
total number of samples as well as the mini-batch size, current
position, etc.  I don't like how this is working right now.

What if we switch it so that a data set describes the actual data set:
how many samples, is it bounded, what data is in it?

Then there is one data stream per role and it tracks how far into the
stream / what the mini-batch size is, etc.  Why is the mini-batch size
a function of hte data stream.  It should just be the total position.

The mini-batch size is a property of the learning algorithm /
execution algorithm - it should propose a mini-batch size and then get
back the actual mini-batch size from the data stream.

The SGB execution context should contain both the mini-batch size.

We need an object like the data stream to be able to request new
sample sequences farther in the future than one step.  Essentially the
stream should be "stateless" about generating indices until they are
consuemed.  Kind of like a future.

Composed of:
 - data reader
 - data stream
 - sample list
 - data packer
