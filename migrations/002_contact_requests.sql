-- Звернення з форми зворотного зв'язку (ТЗ 5.3): асистент відмовив, і форма з'являється одразу під відповіддю.
CREATE TABLE IF NOT EXISTS contact_requests (
    id              BIGSERIAL PRIMARY KEY,
    trace_id        BIGINT REFERENCES traces(id) ON DELETE SET NULL,
    session_id      TEXT,              -- лише якщо користувач погодився прикріпити діалог
    reason          TEXT,              -- outcome відмови: refuse_out_of_scope, refuse_no_info, retracted, error
    question        TEXT NOT NULL,
    product_article TEXT,
    page_url        TEXT,
    contact         TEXT NOT NULL,     -- email або телефон
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS contact_requests_created ON contact_requests (created_at);
