using System;

namespace Zcu.Convsharp.Loaders
{
    [Serializable]
    public abstract class AbstractLoader
    {
        /// <summary>
        /// Variable which contains for batch size
        /// </summary>
        protected int batchSize;
        /// <summary>
        /// Property which contains batch size
        /// </summary>
        public int BatchSize { get { return batchSize; } }
        /// <summary>
        /// Count of batches for training data
        /// </summary>
        protected int trainBatchCount;
        /// <summary>
        /// Property which returns number of 
        /// batches for training data
        /// </summary>
        public int TrainBatchCount { get { return trainBatchCount; } }
        /// <summary>
        /// Count of batches for testing data
        /// </summary>
        protected int testBatchCount;
        /// <summary>
        /// Property which returns number of 
        /// batches for testing data
        /// </summary>
        public int TestBatchCount { get { return testBatchCount; } }
        /// <summary>
        /// Number of items in training dataset
        /// </summary>
        protected int trainItemCount;
        /// <summary>
        /// Property which returns number of 
        /// training items
        /// </summary>
        public int TrainItemCount { get { return trainItemCount; } }
        /// <summary>
        /// Number of items in testing dataset
        /// </summary>
        protected int testItemCount;
        /// <summary>
        /// Into this variable is loaded absolute path to execution directory
        /// </summary>
        protected string currentExecuteDirectory;

        /// <summary>
        /// Method which load full batch with calling load method
        /// </summary>
        /// <param name="batchIndex">idnex of item</param>
        /// <param name="train">variable which is set to true if
        /// we are expecting train data otherwise false</param>
        /// <returns>batch</returns>
        public abstract Tuple<double[][][][], double[][]> LoadBatch(int batchIndex, bool train);
    }
}