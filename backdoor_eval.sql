USE `backdoor_eval`;
/*Table structure for table `ymz_users` */
create table `ymz_users`(
    `u_id` int(8) unsigned zerofill NOT NULL AUTO_INCREMENT,
    `u_name` varchar(6) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `pwd` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    PRIMARY KEY (`u_id`),
    UNIQUE KEY `u_name` (`u_name`)
)ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

/*Table structure for table `ymz_rankings` */
create table `ymz_rankings`(
    `u_id` int(8),
    `m_id` int(8) unsigned zerofill NOT NULL AUTO_INCREMENT,
    `m_name` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `score` float NOT NULL,
    PRIMARY KEY (`m_id`),
    CONSTRAINT `ymz_rankings_ibfk_1` FOREIGN KEY (`u_id`) REFERENCES `ymz_users` (`u_id`)
)ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;



/*Data for the table `ymz_users` */

insert into `ymz_users`(`u_id`,`u_name`,`pwd`) values (00000001,'admin', 'admin');