using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Class which contains method for computing
    /// accuracy for non categorical data.
    /// It can be used in binary cross-entropy
    /// or MSE loss function or in some implemented
    /// in the future.
    /// </summary>
    [Serializable]
    public abstract class AbstractNonCategoricalCostFunction : AbstractCostFunction
    {
        public override double ComputeAccuracy(double[][][][] currentInput, double[][] trainLabels)
        {
            int okCounter = 0;
           
            for (int i = 0; i < currentInput.Length; i++)
            {
                double res =  Math.Abs(GetResult(currentInput[i]) - trainLabels[i][0]);
                if (res == 0)
                {
                    okCounter++;
                }
            }
            
            double correct = Convert.ToDouble(okCounter);
            double all = Convert.ToDouble(currentInput.Length);
            double acc = correct / all;
            return acc;
        }

        public override int GetResult(double[][][] input)
        {
            int b = (int)Math.Round(input[0][0][0], mode: MidpointRounding.AwayFromZero);
            return b;
        }

        public override void TestDimension(double[][][][] input, double[][] targets, int batchStart = 0, int batchEnd = 0)
        {
            int numSamples = input.Length;
            int numClasses = input[0][0][0].Length;
            int exNumSamples;
            if (batchEnd == 0)
                exNumSamples = input.Length;
            else
                exNumSamples = batchEnd - batchStart;
            int exNumClasses = targets[0].Length;

            if (numSamples != exNumSamples
                || numClasses != exNumClasses)
            {
                string msg = "Invalid input data in mean squared error, expected input ("
                    + numSamples + "," + numClasses + ") and found (" + exNumSamples + "," + exNumClasses + ")";
                Utils.ThrowException(msg);
            }

            if (numClasses != 1 || exNumClasses != 1)
            {
                string msg = "Invalid input data in mean squared error, expected input with non categorical data. Which has " +
                    "one output.";
                Utils.ThrowException(msg);
            }
        }
    }
}