using System;

namespace Zcu.Convsharp.Common
{
    /// <summary>
    /// Class for saving results after computing one epoch
    /// </summary>
    [Serializable]
    public class EpochHistory
    {
        /// <summary>
        /// Validation loss - loss on validation (testing set)
        /// is set only if the testing set is add to model
        /// </summary>
        private double valLoss;
        /// <summary>
        /// Training loss - loss on training set
        /// </summary>
        private double trainLoss;
        /// <summary>
        /// Validation accuracy - accuracy on validation (testing set)
        /// is set only if the testing set is add to model
        /// </summary>
        private double valAcc;
        /// <summary>
        /// Training accuracy - accuracy on training set
        /// </summary>
        private double trainAcc;

        /// <summary>
        /// Initializes a new instance of EpochHistory class
        /// </summary>
        /// <param name="valLoss">validation loss</param>
        /// <param name="trainLoss">training loss</param>
        /// <param name="valAcc">validation accuracy</param>
        /// <param name="trainAcc">training accuracy</param>
        public EpochHistory(double valLoss, double trainLoss, double valAcc, double trainAcc)
        {
            this.valLoss = valLoss;
            this.trainLoss = trainLoss;
            this.valAcc = valAcc;
            this.trainAcc = trainAcc;
        }
    }
}