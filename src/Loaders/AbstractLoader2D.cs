using convsharp.Loaders;
using System;

namespace Zcu.Convsharp.Loaders
{
    /// <summary>
    /// Abstract loader which define interface for
    /// class which user have to implement for
    /// loading own dataset
    /// </summary>
    [Serializable]
    public abstract class AbstractLoader2D : AbstractLoader
    {
        /// <summary>
        /// Each child of Loader have to implement
        /// this method for loading item by item
        /// for training and testing data according
        /// to train flag
        /// </summary>
        /// <param name="itemIndex">index of item in loading folder</param>
        /// <param name="train">if is true we expect training data, otherwise testing</param>
        /// <returns>loaded item</returns>
        public abstract Tuple<double[][][], double[]> Load(int itemIndex, bool train);

        /// <summary>
        /// Constructor for creating new instance of loader
        /// </summary>
        /// <param name="trainItemCount">number of training items</param>
        /// <param name="testItemCount">number of testing items</param>
        /// <param name="batchSize">batch size</param>
        public AbstractLoader2D(int trainItemCount, int testItemCount, int batchSize)
        {
            this.batchSize = batchSize;
            this.trainItemCount = trainItemCount;
            this.testItemCount = testItemCount;
            // add one batch which will have different size for
            // the usage of all data
            trainBatchCount = trainItemCount / batchSize;
            if (trainBatchCount * batchSize - trainItemCount != 0)
                trainBatchCount++;
            testBatchCount = testItemCount / batchSize;
            if (testBatchCount * batchSize - testItemCount != 0)
                testBatchCount++;
        }

        /// <summary>
        /// Method which load full batch with calling load method
        /// </summary>
        /// <param name="batchIndex">idnex of item</param>
        /// <param name="train">variable which is set to true if
        /// we are expecting train data otherwise false</param>
        /// <returns>batch</returns>
        public override Tuple<double[][][][], double[][]> LoadBatch(int batchIndex, bool train)
        {
            int max = train ? trainItemCount : testItemCount;
            int from = batchIndex * batchSize;
            int temp = (batchIndex + 1) * batchSize;
            int to = temp;
            if (max < temp)
            {
                to = max;
                batchSize = max - from;
            }
            int counter = 0;

            double[][][][] batchValues = new double[batchSize][][][];
            double[][] batchLabels = new double[batchSize][];

            for (int i = from; i < to; i++)
            {
                var item = Load(i, train);
                batchValues[counter] = item.Item1;
                batchLabels[counter] = item.Item2;
                counter++;
            }

            return Tuple.Create(batchValues, batchLabels);
        }
    }
}