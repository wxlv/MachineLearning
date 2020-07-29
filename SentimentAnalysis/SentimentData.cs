using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    /// <summary>
    /// 情绪数据
    /// </summary>
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText { get; set; }

        [LoadColumn(1),ColumnName("Label")]
        public bool Sentiment { get; set; }
    }
    /// <summary>
    /// 情绪预测
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        /// <summary>
        /// 可能性
        /// </summary>
        public float Probability { get; set; }
        /// <summary>
        /// 得分
        /// </summary>
        public float Score { get; set; }
    }
}
