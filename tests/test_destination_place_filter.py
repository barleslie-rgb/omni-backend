import unittest

from services.destination_engine import _is_destination_attraction_record


class DestinationPlaceFilterTests(unittest.TestCase):
    def test_excludes_requested_local_businesses(self):
        cases = [
            {"name": "Fun Fiesta Cinema", "category": "amenity", "type": "cinema"},
            {"name": "Sunshine Chemist Pharmacy", "category": "amenity", "type": "pharmacy"},
            {"name": "Om Pharmacy", "category": "amenity", "type": "pharmacy"},
            {"name": "Cafe 702", "category": "amenity", "type": "cafe"},
            {"name": "cafe coffee day", "category": "amenity", "type": "cafe"},
            {"name": "Butchers and fishmongers", "category": "shop", "type": "butcher"},
            {"name": "New Durga Supermarket", "category": "shop", "type": "supermarket"},
            {"name": "Janta Supermarket", "category": "shop", "type": "supermarket"},
            {"name": "Brinal's La Gloire Beauty Salon & Academy", "category": "shop", "type": "beauty"},
            {"name": "Vishal's Hair Cutting Salon", "category": "shop", "type": "hairdresser"},
            {"name": "Wednesday market", "category": "amenity", "type": "marketplace"},
            {"name": "informal market", "category": "amenity", "type": "marketplace"},
            {"name": "Mowgli Restaurant", "category": "amenity", "type": "restaurant"},
            {"name": "Jio store", "category": "shop", "type": "mobile_phone"},
            {"name": "Generic Medical Store", "category": "Pharmacy / Chemist"},
        ]
        for record in cases:
            with self.subTest(place=record["name"]):
                self.assertFalse(_is_destination_attraction_record(record))

    def test_keeps_named_attractions(self):
        cases = [
            {"name": "Tungareshwar Waterfalls", "category": "natural", "type": "waterfall"},
            {"name": "Tungareshwar Temple", "category": "amenity", "type": "place_of_worship"},
            {"name": "Vasai Fort", "category": "historic", "type": "fort"},
            {"name": "Jivdani Temple", "category": "amenity", "type": "place_of_worship"},
            {"name": "Arnala Beach", "category": "natural", "type": "beach"},
            {"name": "Taj Mahal", "category": "tourism", "type": "attraction"},
        ]
        for record in cases:
            with self.subTest(place=record["name"]):
                self.assertTrue(_is_destination_attraction_record(record))

    def test_rejects_generic_place_type_names(self):
        for name in ("Temple", "Mosque", "Market", "Park"):
            with self.subTest(place=name):
                self.assertFalse(
                    _is_destination_attraction_record(
                        {"name": name, "category": "amenity", "type": "place_of_worship"}
                    )
                )


if __name__ == "__main__":
    unittest.main()
