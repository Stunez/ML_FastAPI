CREATE SCHEMA qa;

CREATE TABLE qa.qa_data(
	id serial,
	model_name varchar NOT NULL,
	question varchar NOT NULL,
	answer varchar NOT NULL,
	create_date timestamp(0) NOT NULL DEFAULT now()
);