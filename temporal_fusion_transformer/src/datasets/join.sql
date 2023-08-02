SELECT
    temporal.date,
    temporal.open,
    CAST(temporal.onpromotion AS INT),
    temporal.traj_id,
    temporal.unique_id,
    temporal.log_sales,
    EXTRACT(DAYOFWEEK FROM temporal.date)       AS day_of_week,
    EXTRACT(DAY FROM temporal.date)             AS day_of_month,
    EXTRACT(MONTH FROM temporal.date)           AS month,
    stores.*,
    items.*,
    COALESCE(national_holidays.description, "") AS national_hol,
    COALESCE(regional_holidays.description, "") AS regional_hol,
    COALESCE(local_holidays.description, "")    AS local_hol,
    transactions.transactions,
    oil.oil_ffil                                AS oil,

FROM `titanium-atlas-389220.favorita.temporal` AS temporal
    LEFT JOIN
        (
            SELECT date, LAST_VALUE(dcoilwtico IGNORE NULLS) OVER (ORDER BY date) AS oil_ffil
            FROM `titanium-atlas-389220.favorita.oil`
        ) AS oil
            ON temporal.date = oil.date

    LEFT JOIN `titanium-atlas-389220.favorita.stores` AS stores ON temporal.store_nbr = stores.store_nbr
    LEFT JOIN `titanium-atlas-389220.favorita.items` AS items ON temporal.item_nbr = items.item_nbr

    LEFT JOIN `titanium-atlas-389220.favorita.transactions`AS transactions
        ON temporal.date = transactions.date AND temporal.store_nbr = transactions.store_nbr

    LEFT JOIN
        (
            SELECT * FROM `titanium-atlas-389220.favorita.holidays_events`
                WHERE locale = "National"
        ) AS national_holidays
            ON national_holidays.date = temporal.date

    LEFT JOIN
        (
            SELECT * FROM `titanium-atlas-389220.favorita.holidays_events`
                WHERE locale = "Regional"
        ) AS regional_holidays
            ON regional_holidays.date = temporal.date AND regional_holidays.locale_name = stores.state

    LEFT JOIN
        (
            SELECT * FROM `titanium-atlas-389220.favorita.holidays_events`
                WHERE locale = "Local"
        ) AS local_holidays
            ON local_holidays.date = temporal.date AND local_holidays.locale_name = stores.city

WHERE oil.oil_ffil IS NOT NULL AND temporal.traj_id IS NOT NULL AND temporal.log_sales IS NOT NULL