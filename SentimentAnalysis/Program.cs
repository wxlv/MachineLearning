using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    class Program
    {
        //yelp_labelled
        //imdb_labelled
        //yelp_labelled_all
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext();
            TrainTestData splitDataView = LoadData(mLContext);
            ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);
            Evaluate(mLContext, model, splitDataView.TestSet);
            Console.ReadLine();
        }

        public static TrainTestData LoadData(MLContext mLContext)
        {
            //加载训练数据集
            IDataView dataView = mLContext.Data.LoadFromTextFile<SentimentData>(_modelPath, hasHeader: false);
            //切分测试数据集（20%）
            TrainTestData splitData = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitData;
        }
        /// <summary>
        /// 生成和定型模型
        /// </summary>
        /// <param name="mLContext"></param>
        /// <param name="splitTrainSet"></param>
        /// <returns></returns>
        public static ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
        {
            var estimator =
                //提取和转换数据
                mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                //添加学习算法（二分类任务）
                .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }
        /// <summary>
        /// 评估模型
        /// </summary>
        /// <param name="mLContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine("Raw Data: {0}", JsonConvert.SerializeObject(metrics));
            Console.WriteLine($"Accuracy(准确性): {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc(可信度): {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
    }
}
