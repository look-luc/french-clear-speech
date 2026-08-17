DROP TABLE IF EXISTS PUBLIC.DATA_RESPONSE;

CREATE TABLE public.data_response
(
    primary_key serial NOT NULL,
    subject character varying(100) NOT NULL,
    trial_index character varying(30),
    stimulus character varying(128) NOT NULL,
    response text NOT NULL,
    custom_tag character varying(100) DEFAULT 'clear_speech',
    CONSTRAINT pk_data_response PRIMARY KEY (primary_key)
);

ALTER TABLE IF EXISTS public.data_response
    OWNER to lucdenardi;
