import numeric from 'numeric';

export function calculatePredictedDeaths() {
  const years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022];
  const deaths = [601011, 696602, 825837, 825995, 822382, 995835, 1216980, 1136308];

  // 선형 회귀 모델 피팅
  const n = years.length;
  const sumX = numeric.sum(years);
  const sumY = numeric.sum(deaths);
  const sumXY = numeric.dot(years, deaths);
  const sumX2 = numeric.sum(numeric.mul(years, years));
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

  // 오늘 날짜를 기준으로 경과 일 수 계산
  const today = new Date();
  const startOfYear = new Date(today.getFullYear(), 0, 1);
  const daysElapsed = Math.floor((today - startOfYear) / (1000 * 60 * 60 * 24));

  // 예상 증가량을 이용하여 오늘까지의 예상 사망자 수 계산
  const predictedDeathsToday = deaths[deaths.length - 1] + slope * daysElapsed / 365;

  return {
    predictedDeathsToday: Math.floor(predictedDeathsToday)
  };
}