
/*Table structure for table `ymz_users` */
create table `ymz_users`(
    `u_id` int(8) unsigned zerofill NOT NULL AUTO_INCREMENT,
    `u_name` varchar(6) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    `pwd` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
    PRIMARY KEY (`u_id`),
    UNIQUE KEY `username` (`username`)
)AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;