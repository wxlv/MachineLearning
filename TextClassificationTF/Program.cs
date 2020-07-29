using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace TextClassificationTF
{
    class Program
    {
        public const int FeatureLength = 600;
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");
        static void Main(string[] args)
        {
            //初始化ML.NET环境
            MLContext mLContext = new MLContext();
            //创建查找映射
            var lookupMap = mLContext.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
                columns: new[] {
                    new TextLoader.Column("Words",DataKind.String,0),
                    new TextLoader.Column("Ids",DataKind.Int32,1),
                },
                separatorChar: ','
                );
            Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
             {
                 var features = s.VariableLengthFeatures;
                 Array.Resize(ref features, FeatureLength);
                 f.Features = features;
             };
            //添加TensorFlow模型
            TensorFlowModel tensorFlowModel = mLContext.Model.LoadTensorFlowModel(_modelPath);

            #region 模型信息（测试用）
            DataViewSchema schema = tensorFlowModel.GetModelSchema();
            Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
            var featuresType = (VectorDataViewType)schema["Features"].Type;
            Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
            var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;
            Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");
            Console.ReadLine();
            #endregion

            //创建ML.NET管道
            IEstimator<ITransformer> pipeline =
                //使用TokenizeIntoWords分词
                mLContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
                //使用分词结果查找表映射到整数编码
                .Append(mLContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap, lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
                //将可变长度的整数编码，调整为模型需要的固定长度
                .Append(mLContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                //加载TensorFlow模型对输入进行分类
                .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
                //为输出预设创建一个新列
                .Append(mLContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

            //从管道创建ML.NET模型
            IDataView dataView = mLContext.Data.LoadFromEnumerable(new List<MovieReview>());
            ITransformer model = pipeline.Fit(dataView);

            //预测测试模型
            PredictSentiment(mLContext, model);
        }

        public static void PredictSentiment(MLContext mLContext, ITransformer model)
        {
            var engine = mLContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

            var reviews = new List<MovieReview>
            {
                new MovieReview() { ReviewText = "this film is really good" },
                new MovieReview() { ReviewText = "this film is good" },
                new MovieReview() { ReviewText = "this film is fuck" },
                new MovieReview() { ReviewText = "this film is really fuck" },
                new MovieReview() { ReviewText = "this film is fuck good" },
                new MovieReview() { ReviewText = "this film is good fuck" },
                new MovieReview() { ReviewText = "这部电影非常好看" },
                new MovieReview() { ReviewText = "这部电影一般好看" },
                new MovieReview() { ReviewText = "这部电影比较一般，不是很推荐入坑。" }
            };
            reviews.ForEach(item =>
            {
                var sentimentPrediction = engine.Predict(item);
                Console.WriteLine("=============================== {0} ===================================", reviews.IndexOf(item) + 1);
                Console.WriteLine("Raw text: {0}", item.ReviewText);
                Console.WriteLine("Raw data:{0}", JsonConvert.SerializeObject(sentimentPrediction));
                Console.WriteLine("Number of classes: {0}", sentimentPrediction.Prediction.Length);
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Is sentiment/review positive? {0}", sentimentPrediction.Prediction[1] > 0.5 ? "Yes." : "No.");
                Console.ResetColor();
                Console.WriteLine();
            });
            Console.ReadLine();
        }

        public class MovieReview
        {
            public string ReviewText { get; set; }
        }
        public class VariableLength
        {
            [VectorType]
            public int[] VariableLengthFeatures { get; set; }
        }
        public class FixedLength
        {
            [VectorType(FeatureLength)]
            public int[] Features { get; set; }
        }
        /// <summary>
        /// 训练模型后使用的预测类
        /// </summary>
        public class MovieReviewSentimentPrediction
        {
            [VectorType(2)]
            public float[] Prediction { get; set; }
        }
    }


}
