-- MySQL dump 10.13  Distrib 8.0.33, for Win64 (x86_64)
--
-- Host: 43.202.55.53    Database: drug
-- ------------------------------------------------------
-- Server version	11.0.2-MariaDB-1:11.0.2+maria~ubu2204

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ai_url`
--

DROP TABLE IF EXISTS `ai_url`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ai_url` (
  `id` int(11) NOT NULL,
  `url` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ai_url`
--

LOCK TABLES `ai_url` WRITE;
/*!40000 ALTER TABLE `ai_url` DISABLE KEYS */;
INSERT INTO `ai_url` VALUES (1,'https://51af-34-80-42-169.ngrok-free.app');
/*!40000 ALTER TABLE `ai_url` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `content`
--

DROP TABLE IF EXISTS `content`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `content` (
  `content_id` int(11) NOT NULL AUTO_INCREMENT,
  `content_type` varchar(255) DEFAULT NULL,
  `content_like` int(11) NOT NULL DEFAULT 0,
  PRIMARY KEY (`content_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `content`
--

LOCK TABLES `content` WRITE;
/*!40000 ALTER TABLE `content` DISABLE KEYS */;
INSERT INTO `content` VALUES (1,'중독',16),(2,'코카인',15),(3,'필로폰',12);
/*!40000 ALTER TABLE `content` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `content_detail`
--

DROP TABLE IF EXISTS `content_detail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `content_detail` (
  `content_detail_id` int(11) NOT NULL AUTO_INCREMENT,
  `content_id` int(11) NOT NULL,
  `content_order` int(11) NOT NULL,
  `content_img` varchar(255) NOT NULL,
  PRIMARY KEY (`content_detail_id`),
  KEY `FK_content_TO_content_detail_1_idx` (`content_id`),
  CONSTRAINT `FK_content_TO_content_detail_1` FOREIGN KEY (`content_id`) REFERENCES `content` (`content_id`)
) ENGINE=InnoDB AUTO_INCREMENT=43 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `content_detail`
--

LOCK TABLES `content_detail` WRITE;
/*!40000 ALTER TABLE `content_detail` DISABLE KEYS */;
INSERT INTO `content_detail` VALUES (1,1,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/001.png'),(2,1,2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/002.png'),(3,1,3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/003.png'),(4,1,4,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/004.png'),(5,1,5,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/005.png'),(6,1,6,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/addicted/006.png'),(31,2,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/001.png'),(32,2,2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/002.png'),(33,2,3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/003.png'),(34,2,4,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/004.png'),(35,2,5,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/005.png'),(36,2,6,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/cocaine/006.png'),(37,3,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/001.png'),(38,3,2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/002.png'),(39,3,3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/003.png'),(40,3,4,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/004.png'),(41,3,5,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/005.png'),(42,3,6,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/content/meth/006.png');
/*!40000 ALTER TABLE `content_detail` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `content_tag`
--

DROP TABLE IF EXISTS `content_tag`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `content_tag` (
  `content_tag_id` int(11) NOT NULL AUTO_INCREMENT,
  `content_id` int(11) NOT NULL,
  `content_tag_name` varchar(255) NOT NULL,
  PRIMARY KEY (`content_tag_id`),
  KEY `FK_content_TO_content_tag_1_idx` (`content_id`),
  CONSTRAINT `FK_content_TO_content_tag_1` FOREIGN KEY (`content_id`) REFERENCES `content` (`content_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `content_tag`
--

LOCK TABLES `content_tag` WRITE;
/*!40000 ALTER TABLE `content_tag` DISABLE KEYS */;
INSERT INTO `content_tag` VALUES (1,1,'약물정보'),(2,1,'중독'),(3,2,'약물정보'),(4,2,'코카인'),(5,3,'약물정보'),(6,3,'필로폰');
/*!40000 ALTER TABLE `content_tag` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `drug_ba_img`
--

DROP TABLE IF EXISTS `drug_ba_img`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `drug_ba_img` (
  `drug_img_id` int(11) NOT NULL AUTO_INCREMENT,
  `drug_before_img` varchar(255) NOT NULL,
  `drug_after_img` varchar(255) NOT NULL,
  PRIMARY KEY (`drug_img_id`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `drug_ba_img`
--

LOCK TABLES `drug_ba_img` WRITE;
/*!40000 ALTER TABLE `drug_ba_img` DISABLE KEYS */;
INSERT INTO `drug_ba_img` VALUES (1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before1.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after1.PNG'),(2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before2.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after2.PNG'),(3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before3.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after3.PNG'),(4,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before4.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after4.PNG'),(5,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before5.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after5.PNG'),(6,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before6.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after6.PNG'),(7,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before7.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after7.PNG'),(8,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before8.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after8.PNG'),(9,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before9.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after9.PNG'),(10,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before10.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after10.PNG'),(11,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before11.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after11.PNG'),(12,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before12.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after12.PNG'),(13,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before13.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after13.PNG'),(14,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before14.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after14.PNG'),(15,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before15.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after15.PNG'),(16,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/before16.PNG','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/drug/after16.PNG');
/*!40000 ALTER TABLE `drug_ba_img` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `simulation`
--

DROP TABLE IF EXISTS `simulation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `simulation` (
  `simul_no` int(11) NOT NULL,
  `news_url` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`simul_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `simulation`
--

LOCK TABLES `simulation` WRITE;
/*!40000 ALTER TABLE `simulation` DISABLE KEYS */;
INSERT INTO `simulation` VALUES (1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-news.mp4'),(2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/2-news.mp4'),(3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-news.mp4');
/*!40000 ALTER TABLE `simulation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `simulation_detail`
--

DROP TABLE IF EXISTS `simulation_detail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `simulation_detail` (
  `simul_detail_id` int(11) NOT NULL AUTO_INCREMENT,
  `simul_no` int(11) NOT NULL,
  `simul_order` int(11) NOT NULL,
  `simul_content_type` int(11) DEFAULT NULL,
  `simul_video_url` varchar(255) DEFAULT NULL,
  `simul_thumbnail` int(11) DEFAULT NULL,
  `simul_text` varchar(255) DEFAULT NULL,
  `simul_answer1` varchar(255) DEFAULT NULL,
  `simul_answer2` varchar(255) DEFAULT NULL,
  `simul_shift1` int(11) DEFAULT NULL,
  `simul_shift2` int(11) DEFAULT NULL,
  `ar_type` int(11) DEFAULT NULL,
  `source_name` varchar(255) DEFAULT NULL,
  `source_url` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`simul_detail_id`),
  KEY `FK_simulation_TO_simulation_detail_1_idx` (`simul_no`),
  CONSTRAINT `FK_simulation_TO_simulation_detail_1` FOREIGN KEY (`simul_no`) REFERENCES `simulation` (`simul_no`)
) ENGINE=InnoDB AUTO_INCREMENT=36 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `simulation_detail`
--

LOCK TABLES `simulation_detail` WRITE;
/*!40000 ALTER TABLE `simulation_detail` DISABLE KEYS */;
INSERT INTO `simulation_detail` VALUES (1,1,1,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-1.mp4',1,'Chapter 1','친구의 권유',NULL,NULL,NULL,NULL,'(재)한국마약퇴치운동본부','https://www.youtube.com/watch?v=TGbADXb731s'),(2,1,2,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-2.mp4',0,'Chapter 2','투약',NULL,NULL,NULL,NULL,'경기도마약퇴치운동본부','https://www.youtube.com/watch?v=7ADwrxYbG8Y'),(3,1,3,2,NULL,0,'그걸 안하니까 몸이 계속 떨리고, 집중이 되질않는다.\n \n머릿속으로 약이 계속 생각난다.','“아니, 나는 그런거 안해”','“못 참겠어. 딱 한번만 더하자”',2,1,NULL,NULL,NULL),(4,1,4,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-3.mp4',0,'Chapter 3','금단현상',NULL,NULL,NULL,NULL,'범죄예방365','https://www.youtube.com/watch?v=8vLtgMFpxAo'),(5,1,5,3,NULL,0,'하지만 나는 견디지 못했다. \n\n마약의 유혹에서 참을 수 없었다.',NULL,NULL,NULL,NULL,NULL,NULL,NULL),(6,1,6,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-4.mp4',0,'Chapter 4','환각',NULL,NULL,NULL,NULL,NULL,NULL),(7,1,7,4,NULL,0,NULL,NULL,NULL,NULL,NULL,1,NULL,NULL),(8,1,8,2,NULL,0,'그래도 이래선 안되는게 아닐까?','“진짜 마지막으로 딱 한번만...!”',NULL,1,NULL,NULL,NULL,NULL),(9,1,9,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/1-5.mp4',0,'Ending','소년원 이송',NULL,NULL,NULL,NULL,'식품의약품안전처','https://www.youtube.com/watch?v=I9vXzl0s7nI&t=14s'),(18,2,1,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/2-1.mp4',NULL,'Chapter 1','마약 투여',NULL,NULL,NULL,NULL,'JUNE portfolio','https://www.youtube.com/watch?v=4zKJR6g19gI'),(19,2,3,2,NULL,NULL,'마약에 중독되어 살던 중\n임신 사실을 확인했다.\n\n약을 계속하면 아기가 잘못될 수도 있다는데..','아기를 위해 참는다.','너무 힘들어서 견딜 수 없어 ..\n아기에게는 미안하지만 마약을 한다.',1,1,NULL,NULL,NULL),(20,2,4,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/2-2.mp4',1,'Chapter 2','금단증상',NULL,NULL,NULL,NULL,'로지그레이스rosygrace','https://www.youtube.com/watch?v=8ctValeR-gA'),(21,2,5,3,NULL,NULL,'너무 힘든데...\n\n딱 한 번만 더 해볼까?',NULL,NULL,NULL,NULL,NULL,NULL,NULL),(22,2,6,5,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(23,2,7,3,NULL,NULL,'결국 나는 견디지 못했다. \n마약의 유혹에서 참을 수 없었다.\n\n그런데\n태어난 아이가 ..',NULL,NULL,NULL,NULL,NULL,NULL,NULL),(24,2,8,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/2-3.mp4',NULL,'Ending','금단증후군 신생아 ',NULL,NULL,NULL,NULL,'NBC news','https://youtu.be/wL0kulTJR2k'),(25,2,2,4,NULL,NULL,NULL,NULL,NULL,NULL,NULL,2,NULL,NULL),(26,3,1,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-1.mp4',1,'Chapter 1','클럽',NULL,NULL,NULL,NULL,'식품의약품안전처','https://www.youtube.com/watch?v=EX45Ppislfg'),(27,3,2,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-2.mp4',NULL,'Chapter 2','무너지는 생활',NULL,NULL,NULL,NULL,'식품의약품안전처','https://www.youtube.com/watch?v=EX46Ppislfg'),(28,3,3,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-3.mp4',NULL,'Chapter 3','금단현상',NULL,NULL,NULL,NULL,'식품의약품안전처','https://www.youtube.com/watch?v=EX47Ppislfg'),(29,3,4,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-4.mp4',NULL,'Chapter 4','친구의 재권유',NULL,NULL,NULL,NULL,'관세청','https://www.youtube.com/watch?v=99VMWIBJMI8'),(30,3,5,5,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(31,3,6,3,NULL,NULL,'나는 마약의 유혹에서\n참을 수 없었다.',NULL,NULL,NULL,NULL,NULL,NULL,NULL),(32,3,7,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-5.mp4',NULL,'Chapter 5','마약 재투여',NULL,NULL,NULL,NULL,'비주얼텔링 스튜디오','https://www.youtube.com/watch?v=o4Y3PAJehvY'),(33,3,8,3,NULL,NULL,'거울을 바라본다.',NULL,NULL,NULL,NULL,NULL,NULL,NULL),(34,3,9,4,NULL,NULL,NULL,NULL,NULL,NULL,NULL,3,NULL,NULL),(35,3,10,1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/simul/3-6.mp4',NULL,'Ending','죽음(급사)',NULL,NULL,NULL,NULL,'비주얼텔링 스튜디오','https://www.youtube.com/watch?v=o4Y4PAJehvY');
/*!40000 ALTER TABLE `simulation_detail` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `test_answer`
--

DROP TABLE IF EXISTS `test_answer`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_answer` (
  `test_a_id` int(11) NOT NULL AUTO_INCREMENT,
  `test_q_no` int(11) NOT NULL,
  `test_a_content` varchar(255) NOT NULL,
  `test_score` int(11) NOT NULL,
  PRIMARY KEY (`test_a_id`),
  KEY `FK_test_quest_TO_test_answer_1_idx` (`test_q_no`),
  CONSTRAINT `FK_test_quest_TO_test_answer_1` FOREIGN KEY (`test_q_no`) REFERENCES `test_quest` (`test_q_no`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `test_answer`
--

LOCK TABLES `test_answer` WRITE;
/*!40000 ALTER TABLE `test_answer` DISABLE KEYS */;
INSERT INTO `test_answer` VALUES (1,1,'나는 신나는 파티가 좋아',0),(2,1,'나는 대화가 있는 조용한 파티가 좋아',10),(3,2,'나는 자유롭고 쾌활한 친구가 좋아',0),(4,2,'나는 자유 분방하게 행동하는 사람은 싫어',10),(5,3,'이성을 볼 때 외모가 중요해',0),(6,3,'이성을 볼 때 대화 잘 통하는게 중요해',10),(7,4,'가족 영화가 좋아',0),(8,4,'액션 영화가 좋아',10),(9,5,'요즘 영화는 정말 너무 많은\n선정적인 장면을 묘사하고 있다.',0),(10,5,'나는 그 장면을 보는 것이 좋다.',10),(11,6,'나는 무엇을 경험한다는 것 \n그 자체에는 별 관심이 없다.',0),(12,6,'나는 비록 조금은 두렵고, \n관습적이거나 합법적이 아닐지라도\n새롭고 흥분을 자아내는 \n경험과 감각을 맛보고 싶다.',10),(13,7,'지나친 음주는 사람들을 시끄럽고 \n난폭하게 만들어 대체로 파티를 망친다.',0),(14,7,'멋진 파티가 되려면 마실 것이 \n충분히 갖추어져 있어야 한다.',10),(15,8,'나는 한 번도 먹어 본 적이 없는\n새로운 음식을 먹어 보고 싶다.',0),(16,8,'나는 실망할까 봐 내가 잘 아는 \n음식만을 주문한다.',10);
/*!40000 ALTER TABLE `test_answer` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `test_q_result`
--

DROP TABLE IF EXISTS `test_q_result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_q_result` (
  `test_q_result_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `test_a_id` int(11) NOT NULL AUTO_INCREMENT,
  `test_q_no` int(11) NOT NULL,
  PRIMARY KEY (`test_q_result_id`),
  KEY `FK_test_answer_TO_test_q_result_1_idx` (`test_a_id`),
  KEY `FK_test_answer_TO_test_q_result_2_idx` (`test_q_no`),
  KEY `FK_user_TO_test_q_result_1_idx` (`user_id`),
  CONSTRAINT `FK_test_answer_TO_test_q_result_1` FOREIGN KEY (`test_a_id`) REFERENCES `test_answer` (`test_a_id`),
  CONSTRAINT `FK_test_answer_TO_test_q_result_2` FOREIGN KEY (`test_q_no`) REFERENCES `test_answer` (`test_q_no`),
  CONSTRAINT `FK_user_TO_test_q_result_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `test_q_result`
--

LOCK TABLES `test_q_result` WRITE;
/*!40000 ALTER TABLE `test_q_result` DISABLE KEYS */;
/*!40000 ALTER TABLE `test_q_result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `test_quest`
--

DROP TABLE IF EXISTS `test_quest`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_quest` (
  `test_q_no` int(11) NOT NULL,
  `test_q_img` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`test_q_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `test_quest`
--

LOCK TABLES `test_quest` WRITE;
/*!40000 ALTER TABLE `test_quest` DISABLE KEYS */;
INSERT INTO `test_quest` VALUES (1,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q1.gif'),(2,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q2.gif'),(3,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q3.gif'),(4,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q4.gif'),(5,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q5.png'),(6,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q6.gif'),(7,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q7.gif'),(8,'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/Q8.gif');
/*!40000 ALTER TABLE `test_quest` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `test_result`
--

DROP TABLE IF EXISTS `test_result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `test_result` (
  `test_r_id` int(11) NOT NULL AUTO_INCREMENT,
  `test_r_score` int(11) NOT NULL,
  `test_r_title` varchar(255) DEFAULT NULL,
  `test_r_subtitle` varchar(255) DEFAULT NULL,
  `test_r_content` varchar(255) NOT NULL,
  `test_r_img_url` varchar(255) DEFAULT NULL,
  `test_r_count` int(11) DEFAULT NULL,
  PRIMARY KEY (`test_r_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `test_result`
--

LOCK TABLES `test_result` WRITE;
/*!40000 ALTER TABLE `test_result` DISABLE KEYS */;
INSERT INTO `test_result` VALUES (1,0,'말랑말랑 햄스터','스트레스를 받지 않으시나요?','사람들에게 친절한 만큼 주변환경이나 스트레스에 민감한 당신! 좋아하는 것을 계속 찾아가며 취미생활을 늘려 보는 건 어떨까요?','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/QR1.png',60),(2,20,'행복한 강아지','스트레스 수준이 적절합니다.','주변에 휩쓸리지 않고\n자신의 인생을 살고 계시는 군요.\n일탈의 유혹에 넘어가지 않는 모습이\n아주 멋있어요 ~!','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/QR2.png',35),(3,40,'으르렁거리는 퍼그','스트레스가 조금 많습니다.','일상에서 조금 지쳐있는 것 같아요. \n원래 하던 취미나 다른 시간을 가져보고 스트레스를 줄이도록 노력해보도록 해요. 다른 사람을 만나 이야기를 하는 것만으로도 충분히 해소될 수 있어요','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/QR3.png',40),(4,60,'까칠한 고슴도치','스트레스가 너무 높습니다.','갖가지일로 스트레스가 많아요. \n혹시라도 불건전한 일탈에 손을 댈 가능성이 있습니다. 휴식을 가지거나, 잠시 일상과 멀어져보세요','https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/test/QR4.png',84);
/*!40000 ALTER TABLE `test_result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `user_id` int(11) NOT NULL AUTO_INCREMENT,
  `test_final_score` int(11) DEFAULT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=221 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
INSERT INTO `user` VALUES (1,20),(2,40),(3,10),(4,70),(5,75),(6,100),(7,200),(8,200),(9,200),(10,50),(11,60),(12,40),(13,70),(14,70),(15,70),(16,70),(17,70),(18,0),(19,0),(20,0),(21,40),(22,30),(23,70),(24,40),(25,30),(26,30),(27,40),(28,70),(29,30),(30,70),(31,70),(32,60),(33,60),(34,30),(35,40),(36,40),(37,40),(38,40),(39,40),(40,70),(41,70),(42,0),(43,70),(44,70),(45,0),(46,0),(47,0),(48,0),(49,0),(50,0),(51,30),(52,70),(53,70),(54,70),(55,70),(56,70),(57,70),(58,70),(59,70),(60,70),(61,70),(62,40),(63,40),(64,70),(65,70),(66,70),(67,70),(68,70),(69,70),(70,70),(71,70),(72,70),(73,70),(74,70),(75,70),(76,70),(77,70),(78,70),(79,70),(80,70),(81,70),(82,70),(83,70),(84,70),(85,70),(86,70),(87,70),(88,70),(89,70),(90,70),(91,70),(92,70),(93,70),(94,70),(95,70),(96,70),(97,70),(98,70),(99,70),(100,70),(101,70),(102,0),(103,70),(104,0),(105,30),(106,30),(107,70),(108,0),(109,0),(110,0),(111,40),(112,40),(113,40),(114,0),(115,0),(116,40),(117,30),(118,30),(119,30),(120,0),(121,0),(122,0),(123,0),(124,0),(125,0),(126,0),(127,0),(128,0),(129,0),(130,0),(131,0),(132,0),(133,0),(134,0),(135,0),(136,0),(137,70),(138,70),(139,70),(140,70),(141,30),(142,30),(143,70),(144,70),(145,70),(146,0),(147,20),(148,50),(149,10),(150,10),(151,50),(152,50),(153,50),(154,50),(155,50),(156,50),(157,50),(158,50),(159,50),(160,50),(161,50),(162,50),(163,50),(164,50),(165,20),(166,0),(167,0),(168,0),(169,0),(170,0),(171,0),(172,0),(173,20),(174,0),(175,40),(176,0),(177,0),(178,0),(179,60),(180,60),(181,70),(182,30),(183,70),(184,40),(185,40),(186,0),(187,70),(188,10),(189,40),(190,70),(191,30),(192,30),(193,40),(194,30),(195,20),(196,20),(197,0),(198,20),(199,30),(200,60),(201,30),(202,60),(203,30),(204,20),(205,30),(206,30),(207,30),(208,40),(209,40),(210,50),(211,70),(212,0),(213,20),(214,0),(215,0),(216,10),(217,50),(218,0),(219,0),(220,30);
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-08-17 10:04:39
