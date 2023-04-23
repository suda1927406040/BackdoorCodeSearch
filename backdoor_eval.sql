CREATE DATABASE `backdoor_eval`;
USE `backdoor_eval`;
/*Table structure for table `ymz_users` */
create table `ymz_users`(
    `u_id` int(8) unsigned zerofill NOT NULL AUTO_INCREMENT,
    `u_name` varchar(6) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `pwd` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    PRIMARY KEY (`u_id`),
    UNIQUE KEY `u_name` (`u_name`)
)ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

/*Data for the table `ymz_users` */

insert into `ymz_users`(`u_id`,`u_name`,`pwd`) values (00000001, 'admin', 'admin');


/*Table structure for table `ymz_models` */
create table `ymz_models`(
    `m_id` int(8) unsigned zerofill NOT NULL AUTO_INCREMENT,
    `m_name` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `target` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `trigger` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    PRIMARY KEY (`m_id`)
)ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

insert into `ymz_models`(`m_id`, `m_name`,`target`, `trigger`) values (00000001, 'Transformer', 'file', 'wb');

/*Table structure for table `ymz_rankings` */
create table `ymz_rankings`(
    `user_id` int(8),
    `model_id` int(8),
    `score` float NOT NULL,
    PRIMARY KEY (`model_id`, `user_id`),
    CONSTRAINT `ranking_user` FOREIGN KEY (`user_id`) REFERENCES `ymz_users` (`u_id`)
);


/*Data for the table `ymz_rankings` */

insert into `ymz_rankings`(`u_id`, `m_id`,`m_name`, `score`) values (00000001, 00000001, 'Transformer', 4.17);