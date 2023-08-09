
export function calculatePredictedDeaths() {
  const data = [
    { year: 2015, month: "April", deaths: 48748 },
    { year: 2015, month: "August", deaths: 50834 },
    { year: 2015, month: "December", deaths: 52623 },
    { year: 2015, month: "February", deaths: 47725 },
    { year: 2015, month: "January", deaths: 47523 },
    { year: 2015, month: "July", deaths: 50301 },
    { year: 2015, month: "June", deaths: 49691 },
    { year: 2015, month: "March", deaths: 48198 },
    { year: 2015, month: "May", deaths: 49293 },
    { year: 2015, month: "November", deaths: 52386 },
    { year: 2015, month: "October", deaths: 52114 },
    { year: 2015, month: "September", deaths: 51575 },
    { year: 2016, month: "April", deaths: 55763 },
    { year: 2016, month: "August", deaths: 59417 },
    { year: 2016, month: "December", deaths: 63938 },
    { year: 2016, month: "February", deaths: 53834 },
    { year: 2016, month: "January", deaths: 52902 },
    { year: 2016, month: "July", deaths: 58525 },
    { year: 2016, month: "June", deaths: 57428 },
    { year: 2016, month: "March", deaths: 54781 },
    { year: 2016, month: "May", deaths: 56465 },
    { year: 2016, month: "November", deaths: 62340 },
    { year: 2016, month: "October", deaths: 61062 },
    { year: 2016, month: "September", deaths: 60147 },
    { year: 2017, month: "April", deaths: 67493 },
    { year: 2017, month: "August", deaths: 69988 },
    { year: 2017, month: "December", deaths: 70699 },
    { year: 2017, month: "February", deaths: 66189 },
    { year: 2017, month: "January", deaths: 65571 },
    { year: 2017, month: "July", deaths: 69504 },
    { year: 2017, month: "June", deaths: 69153 },
    { year: 2017, month: "March", deaths: 66858 },
    { year: 2017, month: "May", deaths: 68370 },
    { year: 2017, month: "November", deaths: 70723 },
    { year: 2017, month: "October", deaths: 70690 },
    { year: 2017, month: "September", deaths: 70599 },
    { year: 2018, month: "April", deaths: 69042 },
    { year: 2018, month: "August", deaths: 68714 },
    { year: 2018, month: "December", deaths: 67850 },
    { year: 2018, month: "February", deaths: 69745 },
    { year: 2018, month: "January", deaths: 70122 },
    { year: 2018, month: "July", deaths: 68728 },
    { year: 2018, month: "June", deaths: 68714 },
    { year: 2018, month: "March", deaths: 69364 },
    { year: 2018, month: "May", deaths: 68789 },
    { year: 2018, month: "November", deaths: 68102 },
    { year: 2018, month: "October", deaths: 68404 },
    { year: 2018, month: "September", deaths: 68421 },
    { year: 2019, month: "April", deaths: 67736 },
    { year: 2019, month: "August", deaths: 68371 },
    { year: 2019, month: "December", deaths: 71130 },
    { year: 2019, month: "February", deaths: 67631 },
    { year: 2019, month: "January", deaths: 67697 },
    { year: 2019, month: "July", deaths: 68023 },
    { year: 2019, month: "June", deaths: 67787 },
    { year: 2019, month: "March", deaths: 67727 },
    { year: 2019, month: "May", deaths: 67795 },
    { year: 2019, month: "November", deaths: 70357 },
    { year: 2019, month: "October", deaths: 69371 },
    { year: 2019, month: "September", deaths: 68757 },
    { year: 2020, month: "April", deaths: 77017 },
    { year: 2020, month: "August", deaths: 87293 },
    { year: 2020, month: "December", deaths: 92478 },
    { year: 2020, month: "February", deaths: 73343 },
    { year: 2020, month: "January", deaths: 72124 },
    { year: 2020, month: "July", deaths: 85236 },
    { year: 2020, month: "June", deaths: 82916 },
    { year: 2020, month: "March", deaths: 74679 },
    { year: 2020, month: "May", deaths: 80577 },
    { year: 2020, month: "November", deaths: 91200 },
    { year: 2020, month: "October", deaths: 90093 },
    { year: 2020, month: "September", deaths: 88879 },
    { year: 2021, month: "April", deaths: 99772 },
    { year: 2021, month: "August", deaths: 102612 },
    { year: 2021, month: "December", deaths: 107573 },
    { year: 2021, month: "February", deaths: 96118 },
    { year: 2021, month: "January", deaths: 94788 },
    { year: 2021, month: "July", deaths: 101477 },
    { year: 2021, month: "June", deaths: 100569 },
    { year: 2021, month: "March", deaths: 98211 },
    { year: 2021, month: "May", deaths: 99679 },
    { year: 2021, month: "November", deaths: 106554 },
    { year: 2021, month: "October", deaths: 105528 },
    { year: 2021, month: "September", deaths: 104099 },
    { year: 2022, month: "April", deaths: 107720 },
    { year: 2022, month: "August", deaths: 107257 },
    { year: 2022, month: "December", deaths: 108500 },
    { year: 2022, month: "February", deaths: 108606 },
    { year: 2022, month: "January", deaths: 107776 },
    { year: 2022, month: "July", deaths: 107421 },
    { year: 2022, month: "June", deaths: 107263 },
    { year: 2022, month: "March", deaths: 108553 },
    { year: 2022, month: "May", deaths: 107347 },
    { year: 2022, month: "November", deaths: 107942 },
    { year: 2022, month: "October", deaths: 107466 },
    { year: 2022, month: "September", deaths: 107176 },
    { year: 2023, month: "February", deaths: 105258 },
    { year: 2023, month: "January", deaths: 107260 },
  ];

  // 선형 회귀 모델 피팅
  const n = data.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;

  for (const entry of data) {
    sumX += entry.year;
    sumY += entry.deaths;
    sumXY += entry.year * entry.deaths;
    sumX2 += entry.year * entry.year;
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

  // 오늘 날짜를 기준으로 경과 일 수 계산
  const today = new Date();
  const startOfYear = new Date(today.getFullYear(), 0, 1);
  const daysElapsed = Math.floor((today - startOfYear) / (1000 * 60 * 60 * 24));

  // 예상 증가량을 이용하여 오늘까지의 예상 사망자 수 계산
  const predictedDeathsToday = data[data.length - 1].deaths + slope * (daysElapsed / 365);
  return {predictedDeathsToday:Math.floor(predictedDeathsToday)};
}