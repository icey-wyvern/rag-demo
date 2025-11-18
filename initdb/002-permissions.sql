\connect appdb

-- Allow the app role to create LangChain / PGVector tables in the public schema
GRANT USAGE, CREATE ON SCHEMA public TO app_user;

-- Access to existing objects
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public
	GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
	GRANT USAGE, SELECT ON SEQUENCES TO app_user;
